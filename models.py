import torch
from torch import nn, optim
from torch.nn.utils import prune
import pytorch_lightning as pl

from loss import esr_loss, dc_loss, pre_emphasize


class LstmModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        input_size,
        bptt_steps,
        bptt_warmup,
        learn_rate,
        pre_emph_coeffs,
        loss_coeffs,
    ):
        super().__init__()
        self.rec = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.lin = nn.Linear(hidden_size, 1)

        for module, param in self.get_parameters_to_prune():
            prune.identity(module, param)

        self.truncated_bptt_steps = bptt_steps

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LstmModel")
        parser.add_argument("--hidden_size", type=int, default=64)
        parser.add_argument("--input_size", type=int, default=1)
        parser.add_argument("--bptt_steps", type=int, default=2048)
        parser.add_argument("--bptt_warmup", type=int, default=1000),
        parser.add_argument("--learn_rate", type=float, default=5e-3),
        parser.add_argument(
            "--pre_emph_coeffs", type=float, nargs=2, default=[-0.85, 1]
        )
        parser.add_argument("--loss_coeffs", type=float, nargs=2, default=[0.75, 0.25])
        return parent_parser

    @staticmethod
    def from_argparse_args(args):
        return LstmModel(
            args.hidden_size,
            args.input_size,
            args.bptt_steps,
            args.bptt_warmup,
            args.learn_rate,
            args.pre_emph_coeffs,
            args.loss_coeffs,
        )

    def forward(self, x, hiddens=None):
        residual = x[:, :, 0:1]
        x, hiddens = self.rec(x, hiddens)
        return self.lin(x) + residual[:, :, 0:1], hiddens

    def training_step(self, batch, batch_idx, hiddens):
        warmup_step = hiddens is None

        x, y = batch
        y_hat, hiddens = self(x, hiddens)

        if warmup_step:
            # construct dummy loss so that the warmup step does not update parameters
            loss = torch.zeros(1)
            loss.requires_grad_()
        else:
            # in all other steps, calculate actual loss and backpropagate
            loss = self.mixed_loss(y, y_hat)
            self.log("train_loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "hiddens": hiddens}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        self.log("val_loss", self.mixed_loss(y, y_hat), on_step=False, on_epoch=True)

    def mixed_loss(self, y, y_hat):
        # todo: move to loss.py or to a mixin

        coeffs = torch.tensor([[self.hparams.pre_emph_coeffs]], device=self.device)
        y_emph = pre_emphasize(y, coeffs)
        y_hat_emph = pre_emphasize(y_hat, coeffs)

        esr = esr_loss(y_emph, y_hat_emph)
        dc = dc_loss(y, y_hat)

        return self.hparams.loss_coeffs[0] * esr + self.hparams.loss_coeffs[1] * dc

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.learn_rate, weight_decay=1e-4
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 2,
            },
        }

    def tbptt_split_batch(self, batch, split_size):
        warmup = self.hparams.bptt_warmup
        total_steps = batch[0].shape[1]

        splits = [[x[:, :warmup, :] for i, x in enumerate(batch)]]

        for t in range(warmup, total_steps, split_size):
            batch_split = [x[:, t : t + split_size, :] for i, x in enumerate(batch)]
            splits.append(batch_split)

        return splits

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = esr_loss(y, y_hat)
        self.log("test_esrloss", loss)
        return loss, y_hat

    def get_parameters_to_prune(self):
        return (
            (self.rec, "weight_ih_l0"),
            (self.rec, "weight_hh_l0"),
            (self.lin, "weight"),
        )

    def prune(self, amount):
        prune.global_unstructured(
            self.get_parameters_to_prune(),
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

    def _get_live_nodes(self):
        x = torch.rand([1, 2048, self.hparams.input_size]) * 2 - 1
        _, hiddens = self(x)
        return torch.nonzero(torch.abs(hiddens[0]) > 1e-6)[:, -1].squeeze().tolist()

    def get_compression(self):
        return 1 - len(self._get_live_nodes()) / self.hparams.hidden_size

    def compress(self):
        live_nodes = self._get_live_nodes()
        new_hs = len(live_nodes)
        matrix_idx = []
        # 3 for gru, 4 for lstm
        for i in range(4):
            matrix_idx += [live_nodes[j] + i * self.hparams.hidden_size for j in range(new_hs)]

        for module, name in self.get_parameters_to_prune():
            prune.remove(module, name)

        old_state = self.state_dict()
        new_state = {
            "rec.weight_ih_l0": old_state["rec.weight_ih_l0"][matrix_idx],
            "rec.weight_hh_l0": old_state["rec.weight_hh_l0"][matrix_idx][
                :, live_nodes
            ],
            "rec.bias_ih_l0": old_state["rec.bias_ih_l0"][matrix_idx],
            "rec.bias_hh_l0": old_state["rec.bias_hh_l0"][matrix_idx],
            "lin.weight": old_state["lin.weight"][:, live_nodes],
            "lin.bias": old_state["lin.bias"],
        }

        self.rec = nn.LSTM(
            input_size=self.hparams.input_size,
            hidden_size=new_hs,
            num_layers=1,
            batch_first=True,
        )
        self.lin = nn.Linear(new_hs, 1)

        self.load_state_dict(new_state)
