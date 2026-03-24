# BirdCLEF+ 2026 — Autoresearch

This is an experiment to have the LLM autonomously improve an audio classification model for the BirdCLEF+ 2026 Kaggle competition.

## Competition context

- **Task**: Identify 234 wildlife species (birds, amphibians, mammals, reptiles, insects) from audio in Brazil's Pantanal wetlands
- **Metric**: Macro-averaged ROC-AUC (higher is better), skipping classes with no true positives
- **Test data**: ~600 one-minute soundscapes split into 5-second windows
- **Submission**: CPU-only Kaggle notebook (≤90 min, no internet, no GPU)
- **Training data**: ~46K short species recordings + expert-labeled soundscapes
- **Audio format**: 32kHz, ogg

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` or `CLAUDE.md` — repository context
   - `prepare.py` — fixed constants, data loading, dataset, evaluation (ROC-AUC). **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `./data/train_audio/` contains audio files and `./data/train.csv` exists. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on this machine's Apple M4 with MPS (Metal) backend. The training script runs for a **fixed time budget of 10 minutes**. You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture (backbone, custom heads, attention pooling), optimizer (AdamW, SAM, etc.), hyperparameters, augmentations (mixup, SpecAugment, time/freq masking), loss functions (focal loss, asymmetric loss), batch size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and constants.
- Install new packages or add dependencies beyond what's in `requirements.txt`.
- Modify the evaluation metric. `evaluate_rocauc` in `prepare.py` is ground truth.

**The goal is simple: get the highest val_rocauc.** Since the time budget is fixed (10 min), you don't need to worry about training time. Everything is fair game.

**Memory** is a soft constraint. The machine has 16GB unified memory (shared CPU+GPU). Some increase is acceptable for meaningful ROC-AUC gains, but don't OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. A 0.001 improvement from deleting code? Keep.

**The first run**: Your very first run should always be to establish the baseline by running `train.py` as-is.

## Audio-specific ideas to explore

The agent should be creative, but here are starting points:
- **Backbones**: efficientnet_b0/b1/b2, convnext_tiny, resnet34, mobilenetv3
- **Augmentations**: SpecAugment (time/frequency masking), mixup, random gain, noise injection
- **Loss functions**: Focal loss (for class imbalance), asymmetric loss, label smoothing
- **Pooling**: Global average pooling (default), attention pooling, multi-instance learning
- **Learning rate**: Cosine annealing, warmup, discriminative LR (lower for backbone, higher for head)
- **Data**: Use secondary_labels, filter by rating, oversample rare species
- **Mel params**: Different n_mels (64, 128, 256), different hop_length, different fmin/fmax
- **Ensemble tricks**: Test-time augmentation, multi-crop inference

## Output format

Once the script finishes it prints a summary like:

```
---
val_rocauc:       0.850000
training_seconds: 600.1
total_seconds:    650.2
peak_memory_mb:   0.0
num_steps:        1200
num_epochs:       2
num_params_M:     5.2
backbone:         efficientnet_b0
batch_size:       32
```

Extract the key metric:
```
grep "^val_rocauc:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 5 columns:

```
commit	val_rocauc	status	backbone	description
```

1. git commit hash (short, 7 chars)
2. val_rocauc achieved (e.g. 0.850000) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. backbone name
5. short text description of what this experiment tried

Example:

```
commit	val_rocauc	status	backbone	description
a1b2c3d	0.850000	keep	efficientnet_b0	baseline
b2c3d4e	0.865000	keep	efficientnet_b0	added SpecAugment
c3d4e5f	0.840000	discard	convnext_tiny	switched backbone (slower convergence)
d4e5f6g	0.000000	crash	efficientnet_b2	OOM with batch_size 64
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything)
5. Read out the results: `grep "^val_rocauc:\|^peak_memory_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit results.tsv, leave it untracked)
8. If val_rocauc improved (higher), you "advance" the branch, keeping the commit
9. If val_rocauc is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~10 minutes total (+ overhead). If a run exceeds 15 minutes, kill it and treat it as a failure.

**Crashes**: If it's a dumb typo, fix and re-run. If the idea is fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.

## Important notes for inference

Remember: the final submission runs on **CPU-only Kaggle** (no GPU, no internet, ≤90 min for ~600 1-minute soundscapes). Keep this in mind:
- Don't rely on models that are too large for CPU inference
- EfficientNet-B0 infers ~50 5-sec segments/second on CPU — good
- EfficientNet-B3+ might be too slow for 600 × 12 = 7200 segments in 90 min
- Always think about the inference budget alongside training performance
