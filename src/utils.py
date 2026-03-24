"""Utility functions for BirdCLEF+ 2026."""

import random

import numpy as np
import pandas as pd
import torch

from .config import SAMPLE_SUBMISSION, TAXONOMY_CSV


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_label_mapping() -> tuple[dict, dict]:
    """Build label-to-index and index-to-label mappings from taxonomy.csv."""
    taxonomy = pd.read_csv(TAXONOMY_CSV)
    labels = sorted(taxonomy["primary_label"].tolist())
    label2idx = {label: idx for idx, label in enumerate(labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label


def get_submission_columns() -> list[str]:
    """Get the species column names from the sample submission."""
    sample = pd.read_csv(SAMPLE_SUBMISSION, nrows=1)
    return [c for c in sample.columns if c != "row_id"]
