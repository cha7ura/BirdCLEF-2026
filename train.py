"""
BirdCLEF+ 2026 — Training script. Single-device (MPS/CPU), single-file.
Adapted from Karpathy's autoresearch pattern for audio classification.

Usage: python train.py

THE AGENT MODIFIES THIS FILE. Everything is fair game:
model architecture, optimizer, hyperparameters, augmentation, batch size, etc.
"""

import gc
import math
import time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from prepare import (
    TIME_BUDGET,
    NUM_CLASSES,
    N_MELS,
    SEGMENT_SAMPLES,
    HOP_LENGTH,
    load_train_df,
    get_label2idx,
    make_dataloader,
    evaluate_rocauc,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BirdCLEFModel(nn.Module):
    """Mel-spectrogram classifier using a pretrained CNN backbone."""

    def __init__(self, backbone_name="efficientnet_b0", num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,   # remove classifier head
        )
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, N_MELS, T)
        features = self.backbone(x)
        return self.head(features)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Model
BACKBONE = "efficientnet_b0"
PRETRAINED = True

# Optimization
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER = "cosine"          # "cosine" or "none"
WARMUP_STEPS = 50

# Data
NUM_WORKERS = 4
VAL_RATIO = 0.15

# Loss
LABEL_SMOOTHING = 0.0
POS_WEIGHT = None             # set to a float for weighted BCE, or None

# Seed
SEED = 42

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device selection: MPS > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# Data
label2idx = get_label2idx()
train_df, val_df = load_train_df(val_ratio=VAL_RATIO, seed=SEED)
print(f"Train: {len(train_df):,} samples | Val: {len(val_df):,} samples")

train_loader = make_dataloader(
    train_df, label2idx, batch_size=BATCH_SIZE,
    random_crop=True, num_workers=NUM_WORKERS, shuffle=True,
)
val_loader = make_dataloader(
    val_df, label2idx, batch_size=BATCH_SIZE,
    random_crop=False, num_workers=NUM_WORKERS, shuffle=False,
)

# Model
model = BirdCLEFModel(backbone_name=BACKBONE, pretrained=PRETRAINED).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {BACKBONE} | Params: {num_params / 1e6:.1f}M")

# Loss
if POS_WEIGHT is not None:
    pw = torch.full((NUM_CLASSES,), POS_WEIGHT, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
else:
    criterion = nn.BCEWithLogitsLoss(label_smoothing=LABEL_SMOOTHING) if hasattr(nn.BCEWithLogitsLoss, 'label_smoothing') else nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# LR scheduler
# Estimate total steps: TIME_BUDGET / avg_step_time. ~230ms on MPS, ~80ms on CUDA
avg_step_ms = 100 if torch.cuda.is_available() else 250
total_steps_estimate = int(TIME_BUDGET / (avg_step_ms / 1000))
if SCHEDULER == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_estimate)
else:
    scheduler = None

print(f"Time budget: {TIME_BUDGET}s")
print(f"Batch size: {BATCH_SIZE}")
print(f"LR: {LEARNING_RATE}")
print()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0
step = 0
epoch = 0
smooth_loss = 0

model.train()

while True:
    epoch += 1
    for batch_idx, (mel, target) in enumerate(train_loader):
        t0 = time.time()

        mel = mel.to(device)
        target = target.to(device)

        # Forward
        logits = model(mel)
        loss = criterion(logits, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        t1 = time.time()
        dt = t1 - t0

        # Skip first few steps (warmup / compilation overhead)
        if step > 5:
            total_training_time += dt

        loss_val = loss.item()

        # Fast fail
        if math.isnan(loss_val) or loss_val > 100:
            print("FAIL: loss exploded")
            exit(1)

        # Logging
        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0, TIME_BUDGET - total_training_time)

        if step % 20 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"\rstep {step:05d} ({100*progress:.1f}%) | loss: {debiased_loss:.4f} | lr: {lr_now:.6f} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        step += 1

        # Time's up
        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()
print(f"Training complete: {step} steps, {epoch} epochs")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("Evaluating...")
val_rocauc, per_class_auc = evaluate_rocauc(model, val_loader, device)

# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

t_end = time.time()

# Memory tracking
if device.type == "mps":
    # MPS doesn't have max_memory_allocated, use a rough estimate
    peak_memory_mb = 0.0
elif device.type == "cuda":
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_memory_mb = 0.0

print("---")
print(f"val_rocauc:       {val_rocauc:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_memory_mb:   {peak_memory_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_epochs:       {epoch}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"backbone:         {BACKBONE}")
print(f"batch_size:       {BATCH_SIZE}")

# Top/bottom per-class AUCs
if per_class_auc:
    sorted_auc = sorted(per_class_auc.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 species AUC:")
    for name, auc in sorted_auc[:5]:
        print(f"  {name:30s} {auc:.4f}")
    print(f"Bottom 5 species AUC:")
    for name, auc in sorted_auc[-5:]:
        print(f"  {name:30s} {auc:.4f}")
    print(f"Classes with AUC > 0.5: {sum(1 for v in per_class_auc.values() if v > 0.5)}/{len(per_class_auc)}")
