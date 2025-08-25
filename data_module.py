import torch
import soundfile as sf
from typing import Union
from pathlib import Path
import numpy as np
import random
import pytorch_lightning as pl
import torch.utils
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_src_dir: Union[str, Path], data_tgt_dir: Union[str, Path], is_train: bool, cut_len: int = 16000 * 4):
        self.is_train = is_train
        self.cut_len = cut_len
        self.data_src_dir = Path(data_src_dir)
        self.data_tgt_dir = Path(data_tgt_dir)
        self.wav_names = [p.stem for p in self.data_src_dir.glob('*.wav')]
    
    def normalize_src_tgt(self, src, tgt, eps=1e-8):
        # use reference channel (first channel) to normalize all channel
        # src: [num_mics, T], tgt: [num_mics, T]
        norm_factor = src[0, :].std() + eps
        src = src / norm_factor
        tgt = tgt / norm_factor
        return src, tgt

    def load_wav(self, path):
        wav, sr = sf.read(path, dtype='float32')
        assert sr == 16000
        if wav.ndim == 1:
            return wav[None, :]
        elif wav.ndim == 2:
            return wav.T
        else:
            raise ValueError('wav ndim > 2')

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        name = self.wav_names[idx]
        src = self.load_wav(self.data_src_dir / (name + '.wav'))
        tgt = self.load_wav(self.data_tgt_dir / (name + '.wav'))

        length = src.shape[-1]
        assert length == tgt.shape[-1]

        if not self.is_train:
            tgt = tgt[0]  # [length,], reference
            src, tgt = self.normalize_src_tgt(src, tgt)
            return src, tgt, length, name

        # randomly permute the channel index
        indices = torch.randperm(src.shape[0]).numpy()
        src = src[indices]
        tgt = tgt[indices]
        tgt = tgt[0]  # [length,], reference

        if length < self.cut_len:
            src = np.pad(src, ((0, 0), (0, self.cut_len - length)), mode='wrap')
            tgt = np.pad(tgt, (0, self.cut_len - length), mode='wrap')
        else:
            # randomly cut segment
            wav_start = random.randint(0, length - self.cut_len)
            src = src[:, wav_start: wav_start + self.cut_len]
            tgt = tgt[wav_start: wav_start + self.cut_len]
        
        src, tgt = self.normalize_src_tgt(src, tgt)
        length = self.cut_len
        return src, tgt, length, name


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_src_dir,
        train_tgt_dir,
        val_src_dir,
        val_tgt_dir,
        test_src_dir,
        test_tgt_dir,
        batch_size, 
        cut_len, 
        num_workers,
    ):
        super().__init__()
        self.train_src_dir = train_src_dir
        self.train_tgt_dir = train_tgt_dir
        self.val_src_dir = val_src_dir
        self.val_tgt_dir = val_tgt_dir
        self.test_src_dir = test_src_dir
        self.test_tgt_dir = test_tgt_dir

        self.batch_size = batch_size
        self.cut_len = cut_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(self.train_src_dir, self.train_tgt_dir, is_train=True, cut_len=self.cut_len)
            self.val_dataset = Dataset(self.val_src_dir, self.val_tgt_dir, is_train=False, cut_len=self.cut_len)
        if stage == 'test' or stage is None:
            self.test_dataset = Dataset(self.test_src_dir, self.test_tgt_dir, is_train=False, cut_len=self.cut_len)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False)