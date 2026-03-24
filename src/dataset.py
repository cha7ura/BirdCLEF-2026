"""Dataset classes for BirdCLEF+ 2026."""

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from .config import (
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    TRAIN_AUDIO_DIR,
)


class BirdCLEFDataset(Dataset):
    """Dataset for training on individual species recordings (train_audio)."""

    def __init__(self, df: pd.DataFrame, label2idx: dict, transform=None):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.num_classes = len(label2idx)
        self.transform = transform
        self.target_samples = SAMPLE_RATE * SEGMENT_DURATION
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = TRAIN_AUDIO_DIR / row["filename"]

        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Random crop or pad to target length
        if waveform.shape[1] > self.target_samples:
            start = np.random.randint(0, waveform.shape[1] - self.target_samples)
            waveform = waveform[:, start : start + self.target_samples]
        else:
            pad = self.target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # Mel spectrogram
        mel = self.mel_spec(waveform)
        mel = self.db_transform(mel)

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        if self.transform:
            mel = self.transform(mel)

        # Multi-hot target
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if row["primary_label"] in self.label2idx:
            target[self.label2idx[row["primary_label"]]] = 1.0

        return mel, target
