import os
from data import AudioDataModule
from models import LstmModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
import json

from argparse import ArgumentParser


class MaskSearchEarlyStopping(EarlyStopping):

    # Implements the "early-bird ticket" stopping criterion
    # https://arxiv.org/abs/1909.11957
    # Stop training once the distance between successive pruning masks falls below some threshold

    def __init__(self, pl_module, prune_amount, k_distances=5, **kwargs):
        super().__init__(**kwargs)
        self.k_distances = k_distances
        self.last_distances = []
        self.prune_amount = prune_amount

        self.currently_alive = {}
        for model, param in pl_module.get_parameters_to_prune():
            self.currently_alive[(model, param)] = torch.nonzero(
                model.state_dict()[param + "_mask"], as_tuple=True
            )

        self.prev_mask = self._get_pruning_mask(pl_module)

    def on_validation_end(self, trainer, pl_module):
        pass

    def _get_pruning_mask(self, pl_module):
        # save current state
        state_dict = pl_module.state_dict()

        # retrieve the mask that would result from pruning the model at the current state
        pl_module.prune(self.prune_amount)
        mask = {}
        for model, param in pl_module.get_parameters_to_prune():
            mask[(model, param)] = model.state_dict()[param + "_mask"][
                self.currently_alive[(model, param)]
            ].cpu()

        # undo pruning, we only want the mask
        pl_module.load_state_dict(state_dict)

        return mask

    def on_train_epoch_end(self, trainer, pl_module):

        curr_mask = self._get_pruning_mask(pl_module)

        # calculate maximal hamming distance between current and previous mask
        total_params, total_distance = 0, 0
        for model, param in pl_module.get_parameters_to_prune():
            key = (model, param)
            total_distance += torch.abs(self.prev_mask[key] - curr_mask[key]).sum()
            total_params += curr_mask[key].numel()

        self.last_distances.append(total_distance / total_params)

        # stopping criterion is last k mask distances being less than a threshold
        if len(self.last_distances) == self.k_distances:
            distance = max(self.last_distances)
            self.last_distances.pop(0)
        else:
            distance = 1.0

        self.prev_mask = curr_mask
        pl_module.log("mask_distance", distance)
        self._run_early_stopping_check(trainer)


def mask_search(args):
    base_path = (
        f"{args.device_name}-LSTM{args.hidden_size}-prune{int(args.prune_amount * 100)}"
    )
    data = AudioDataModule.from_argparse_args(args)
    model = LstmModel.from_argparse_args(args)

    meta_logs = {"compression": [], "val_loss": []}

    for i in range(args.prune_iters):
        dir_path = os.path.join(base_path, str(i))

        ckpt_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1)
        callbacks = [
            ckpt_callback,
            MaskSearchEarlyStopping(
                model,
                prune_amount=args.prune_amount,
                monitor="mask_distance",
                patience=10,
                stopping_threshold=0.1,
                mode="min",
                verbose=True,
            ),
        ]

        # constructing a new trainer resets optimizers and schedulers
        trainer = Trainer(
            default_root_dir=dir_path,
            enable_progress_bar=False,
            callbacks=callbacks,
            gpus=1,
            log_every_n_steps=1,
            check_val_every_n_epoch=2,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, data)
        model = LstmModel.load_from_checkpoint(ckpt_callback.best_model_path)

        meta_logs["compression"].append(model.get_compression())
        meta_logs["val_loss"].append(ckpt_callback.best_model_score.item())

        model.prune(args.prune_amount)

    with open(os.path.join(base_path, "meta_logs.json"), "w") as fp:
        json.dump(meta_logs, fp)


if __name__ == "__main__":
    parser = ArgumentParser()
    AudioDataModule.add_argparse_args(parser)
    parser = LstmModel.add_model_specific_args(parser)

    parser.add_argument("--prune_iters", type=int, default=15)
    parser.add_argument("--prune_amount", type=float, default=0.3)

    mask_search(parser.parse_args())
