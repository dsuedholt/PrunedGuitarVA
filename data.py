import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import pytorch_lightning as pl


class AudioDataset(Dataset):
    def __init__(
        self,
        input_file: str = None,
        target_file: str = None,
        segment_length_seconds: float = 0.5,
    ):
        self.input, self.frame_rate = torchaudio.load(input_file, channels_first=False)
        total_frames, num_channels = self.input.shape

        self.target, target_frame_rate = torchaudio.load(
            target_file, channels_first=False
        )

        assert self.frame_rate == target_frame_rate
        assert self.input.shape == self.target.shape

        self.segment_length_samples = int(segment_length_seconds * self.frame_rate)

        if self.segment_length_samples == 0:
            self.segment_length_samples = total_frames

        self.num_segments = int(total_frames / self.segment_length_samples)

    def __getitem__(self, index):
        start = index * self.segment_length_samples
        stop = (index + 1) * self.segment_length_samples
        return self.input[start:stop, :], self.target[start:stop, :]

    def __len__(self):
        return self.num_segments


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        device_name: str = "ht1",
        data_dir: str = "Data",
        segment_length: float = 0.5,
        input_ext: str = "input.wav",
        target_ext: str = "target.wav",
        batch_size: int = 40,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.input_file = device_name + "-" + input_ext
        self.target_file = device_name + "-" + target_ext
        self.segment_length = segment_length
        self.batch_size = batch_size
        self.datasets = {}

    def setup(self, stage):
        def make_dataset(split, segment_length):
            return AudioDataset(
                os.path.join(self.data_dir, split, self.input_file),
                os.path.join(self.data_dir, split, self.target_file),
                segment_length,
            )

        self.datasets["train"] = make_dataset("train", self.segment_length)
        self.datasets["val"] = make_dataset("val", 0)
        self.datasets["test"] = make_dataset("test", 0)

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.datasets["val"])

    def test_dataloader(self):
        return DataLoader(self.datasets["test"])
