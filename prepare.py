"""
BirdCLEF+ 2026 — Data preparation and evaluation utilities.
Adapted from Karpathy's autoresearch pattern for audio classification.

This file is FIXED. The agent does NOT modify it.

Usage:
    python prepare.py              # unzip data, verify structure
    python prepare.py --eda        # also print dataset statistics

Data is expected at ./data/ (downloaded via `kaggle competitions download -c birdclef-2026 -p data/`)
"""

import os
import sys
import time
import zipfile
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 32000          # all audio resampled to 32kHz
SEGMENT_DURATION = 5         # seconds per prediction window
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION  # 160000
NUM_CLASSES = 234            # species to predict
TIME_BUDGET = 600            # training time budget in seconds (10 minutes)

# Mel spectrogram parameters
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 50
FMAX = 14000

# Evaluation
EVAL_MAX_SAMPLES = 2000      # max validation samples for eval (speed)

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_AUDIO_DIR = DATA_DIR / "train_audio"
TRAIN_SOUNDSCAPES_DIR = DATA_DIR / "train_soundscapes"
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_SOUNDSCAPES_LABELS = DATA_DIR / "train_soundscapes_labels.csv"
TAXONOMY_CSV = DATA_DIR / "taxonomy.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def unzip_data():
    """Unzip competition data if not already extracted."""
    zip_path = DATA_DIR / "birdclef-2026.zip"
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found. Download with:")
        print(f"  kaggle competitions download -c birdclef-2026 -p {DATA_DIR}")
        sys.exit(1)

    # Check if already extracted
    if TRAIN_CSV.exists() and TAXONOMY_CSV.exists():
        print("Data already extracted.")
        return

    print(f"Extracting {zip_path} ...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    t1 = time.time()
    print(f"Extracted in {t1 - t0:.1f}s")


def verify_data():
    """Verify all expected files/dirs exist."""
    required = [TRAIN_CSV, TAXONOMY_CSV, SAMPLE_SUBMISSION]
    required_dirs = [TRAIN_AUDIO_DIR]
    for p in required:
        assert p.exists(), f"Missing: {p}"
    for d in required_dirs:
        assert d.exists(), f"Missing directory: {d}"
    print("Data verification passed.")


# ---------------------------------------------------------------------------
# Label mapping (from taxonomy.csv)
# ---------------------------------------------------------------------------

_label2idx = None
_idx2label = None
_species_list = None


def _load_taxonomy():
    global _label2idx, _idx2label, _species_list
    if _label2idx is not None:
        return
    taxonomy = pd.read_csv(TAXONOMY_CSV)
    _species_list = taxonomy["primary_label"].tolist()
    _label2idx = {label: idx for idx, label in enumerate(_species_list)}
    _idx2label = {idx: label for label, idx in _label2idx.items()}


def get_label2idx():
    _load_taxonomy()
    return _label2idx


def get_idx2label():
    _load_taxonomy()
    return _idx2label


def get_species_list():
    """Return species list in submission column order."""
    _load_taxonomy()
    return _species_list


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def load_train_df(val_ratio=0.15, seed=42):
    """
    Load train.csv and split into train/val DataFrames.
    Stratified by primary_label so every species appears in both splits.
    Returns (train_df, val_df).
    """
    df = pd.read_csv(TRAIN_CSV)

    # Only keep rows whose audio files actually exist
    df["filepath"] = df["filename"].apply(lambda f: TRAIN_AUDIO_DIR / f)
    df = df[df["filepath"].apply(lambda p: p.exists())].reset_index(drop=True)

    # Only keep species that are in the taxonomy (submission columns)
    label2idx = get_label2idx()
    df = df[df["primary_label"].isin(label2idx)].reset_index(drop=True)

    # Stratified split
    rng = np.random.RandomState(seed)
    val_indices = []
    train_indices = []
    for label, group in df.groupby("primary_label"):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_indices.extend(idx[:n_val])
        train_indices.extend(idx[n_val:])

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    return train_df, val_df


def load_soundscape_labels():
    """
    Load train_soundscapes_labels.csv if it exists.
    Returns DataFrame with columns: filename, start, end, primary_label (list of species).
    """
    if not TRAIN_SOUNDSCAPES_LABELS.exists():
        return None
    df = pd.read_csv(TRAIN_SOUNDSCAPES_LABELS)
    return df


# ---------------------------------------------------------------------------
# Audio loading utilities
# ---------------------------------------------------------------------------

def load_audio_segment(filepath, target_sr=SAMPLE_RATE, duration=SEGMENT_DURATION, random_crop=True):
    """
    Load an audio file, resample, convert to mono, and crop/pad to fixed duration.
    Returns tensor of shape (1, SEGMENT_SAMPLES).
    """
    waveform, sr = torchaudio.load(filepath)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_samples = target_sr * duration

    # Random crop or pad
    if waveform.shape[1] > target_samples:
        if random_crop:
            start = torch.randint(0, waveform.shape[1] - target_samples, (1,)).item()
        else:
            start = 0
        waveform = waveform[:, start:start + target_samples]
    else:
        pad = target_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    return waveform


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BirdCLEFDataset(torch.utils.data.Dataset):
    """
    Dataset for BirdCLEF train_audio recordings.
    Returns (mel_spectrogram, multi_hot_target).
    mel_spectrogram: shape (1, N_MELS, time_frames) — ready for CNN input.
    """

    def __init__(self, df, label2idx, random_crop=True):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.num_classes = len(label2idx)
        self.random_crop = random_crop
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
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
        filepath = TRAIN_AUDIO_DIR / row["filename"]

        try:
            waveform = load_audio_segment(filepath, random_crop=self.random_crop)
        except Exception:
            # Return silence + zero target on failure
            waveform = torch.zeros(1, SEGMENT_SAMPLES)

        # Mel spectrogram
        mel = self.mel_transform(waveform)        # (1, N_MELS, T)
        mel = self.db_transform(mel)
        # Normalize per-sample
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # Multi-hot target
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        primary = row["primary_label"]
        if primary in self.label2idx:
            target[self.label2idx[primary]] = 1.0

        # Also encode secondary labels if available
        if "secondary_labels" in row and pd.notna(row["secondary_labels"]):
            try:
                sec = eval(row["secondary_labels"]) if isinstance(row["secondary_labels"], str) else []
                for s in sec:
                    if s in self.label2idx:
                        target[self.label2idx[s]] = 1.0
            except Exception:
                pass

        return mel, target


def make_dataloader(df, label2idx, batch_size, random_crop=True, num_workers=4, shuffle=True):
    """Create a DataLoader from a DataFrame."""
    dataset = BirdCLEFDataset(df, label2idx, random_crop=random_crop)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_rocauc(model, val_loader, device, max_samples=EVAL_MAX_SAMPLES):
    """
    Compute macro-averaged ROC-AUC, skipping classes with no true positives.
    This mirrors the competition metric.
    Returns (rocauc, per_class_auc_dict).
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_preds = []
    all_targets = []
    n_samples = 0

    with torch.no_grad():
        for mel, target in val_loader:
            mel = mel.to(device)
            logits = model(mel)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(target.numpy())
            n_samples += mel.shape[0]
            if n_samples >= max_samples:
                break

    all_preds = np.concatenate(all_preds, axis=0)[:max_samples]
    all_targets = np.concatenate(all_targets, axis=0)[:max_samples]

    # Macro ROC-AUC, skipping classes with no positive samples
    aucs = []
    species_list = get_species_list()
    per_class = {}
    for i in range(all_targets.shape[1]):
        if all_targets[:, i].sum() > 0:
            try:
                auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                aucs.append(auc)
                per_class[species_list[i]] = auc
            except ValueError:
                pass

    macro_auc = np.mean(aucs) if aucs else 0.0
    return macro_auc, per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BirdCLEF 2026 data")
    parser.add_argument("--eda", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    print()

    # Step 1: Unzip
    unzip_data()
    print()

    # Step 2: Verify
    verify_data()
    print()

    # Step 3: Load and summarize
    train_df, val_df = load_train_df()
    label2idx = get_label2idx()
    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples:   {len(val_df):,}")
    print(f"Species:       {len(label2idx)}")
    print()

    if args.eda:
        taxonomy = pd.read_csv(TAXONOMY_CSV)
        print("=== Taxonomy ===")
        if "class_name" in taxonomy.columns:
            print(taxonomy["class_name"].value_counts().to_string())
        print()

        print("=== Train label distribution (top 20) ===")
        print(train_df["primary_label"].value_counts().head(20).to_string())
        print()

        print("=== Train label distribution (bottom 20) ===")
        print(train_df["primary_label"].value_counts().tail(20).to_string())
        print()

        if "rating" in train_df.columns:
            print("=== Rating distribution ===")
            print(train_df["rating"].describe().to_string())
            print()

        soundscape_labels = load_soundscape_labels()
        if soundscape_labels is not None:
            print(f"=== Soundscape labels: {len(soundscape_labels):,} segments ===")
            print()

    print("Done! Ready to train.")
