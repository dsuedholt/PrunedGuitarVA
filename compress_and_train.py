from models import LstmModel
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from data import AudioDataModule

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def compress_and_train(args):
    data = AudioDataModule.from_argparse_args(args)

    callbacks = [
        ModelCheckpoint(monitor='val_loss', save_top_k=1, save_last=True),
        EarlyStopping(monitor='val_loss')
    ]
    trainer = Trainer(dirpath=args.dirpath, gpus=1, check_val_every_n_epoch=2, enable_progress_bar=False, num_sanity_val_steps=0, callbacks=callbacks)
    model = LstmModel.load_from_checkpoint(args.ckpt)
    model.compress()

    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path='best')


if __name__ == "__main__":
    # todo: construct ckpt_path from device, architecture, prune_pct, prune_iter
    parser = ArgumentParser()
    AudioDataModule.add_argparse_args(parser)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--dirpath", type=str, default=".")

    compress_and_train(parser.parse_args())
