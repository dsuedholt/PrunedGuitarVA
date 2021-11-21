import os
from models import LstmModel
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from data import AudioDataModule
from utils import latest_checkpoint_from_folder

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def compress_and_train(args):
    dev = args.device_name
    hs = args.hidden_size
    prune_pct = args.prune_pct
    prune_iter = args.prune_iter

    base_dir = f"{dev}-LSTM{hs}-prune{prune_pct}"
    base_dir = os.path.join(base_dir, str(prune_iter))

    ckpt_path = latest_checkpoint_from_folder(base_dir)

    data = AudioDataModule.from_argparse_args(args)

    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True),
        EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience),
    ]

    dir_path = os.path.join(base_dir, "completed")
    trainer = Trainer(
        default_root_dir=dir_path,
        gpus=args.num_gpus,
        check_val_every_n_epoch=2,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    model = LstmModel.load_from_checkpoint(ckpt_path)
    model.compress()

    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser()
    AudioDataModule.add_argparse_args(parser)
    parser.add_argument("--prune_pct", type=int, default=30)
    parser.add_argument("--prune_iter", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--num_gpus", type=int, default=1)

    compress_and_train(parser.parse_args())
