# BirdCLEF+ 2026 — Agent Setup Guide

Complete guide for a Claude Code agent to set up and run autonomous experiments on a fresh machine.

## 1. Competition Overview

- **Competition**: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
- **Task**: Multi-label classification — identify 234 wildlife species (birds, amphibians, mammals, reptiles, insects) from audio recordings in Brazil's Pantanal wetlands
- **Metric**: Macro-averaged ROC-AUC (higher is better), skipping classes with no true positives
- **Submission**: CPU-only Kaggle notebook, ≤90 min runtime, no internet, no GPU
- **Deadline**: June 3, 2026
- **Prize**: $50,000 total

## 2. Repository Structure

```
BirdCLEF/
├── prepare.py          # FIXED — data loading, dataset, evaluation (ROC-AUC). DO NOT MODIFY.
├── train.py            # AGENT MODIFIES THIS — model architecture, optimizer, training loop
├── program.md          # Autonomous experiment loop instructions (read this first)
├── CLAUDE.md           # Project context for Claude Code
├── requirements.txt    # Python dependencies
├── .gitignore
├── data/               # Competition data (gitignored, must download)
│   ├── train_audio/    # ~46K short species recordings (32kHz ogg)
│   ├── train_soundscapes/  # 1-min field recordings from same sites as test
│   ├── train.csv       # Metadata for train_audio
│   ├── train_soundscapes_labels.csv  # Expert annotations (5-sec segments)
│   ├── taxonomy.csv    # 234 species (defines submission columns)
│   └── sample_submission.csv
├── run.log             # Training output (gitignored)
└── results.tsv         # Experiment log (gitignored)
```

## 3. Machine Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU recommended (CUDA). Also works on Apple MPS or CPU.
- ~20GB free disk space (16GB data + model weights)
- Kaggle CLI configured (`~/.kaggle/kaggle.json`)

### Step-by-step

```bash
# 1. Clone the repo
git clone https://github.com/cha7ura/BirdCLEF-2026.git
cd BirdCLEF-2026

# 2. Install dependencies
pip install torch torchvision torchaudio  # use appropriate CUDA version for your GPU
pip install timm scikit-learn pandas numpy matplotlib audiomentations librosa soundfile tqdm

# 3. Download competition data (~16GB)
kaggle competitions download -c birdclef-2026 -p data/

# 4. Extract data and verify
python prepare.py --eda

# Expected output:
#   Train samples: 30,295
#   Val samples:   5,254
#   Species:       234
#   Done! Ready to train.

# 5. Run baseline to verify everything works
python train.py > run.log 2>&1
grep "^val_rocauc:" run.log
```

### Platform-specific notes

**NVIDIA GPU (recommended)**:
- No changes needed — `train.py` auto-detects CUDA
- Batch size 64-128 should work on 8GB+ VRAM
- Expected: ~50-100ms/step with EfficientNet-B0

**Apple MPS**:
- `train.py` already has `multiprocessing.set_start_method("fork")` for macOS
- Batch size 32 for 16GB unified memory
- Expected: ~230ms/step

**CPU only**:
- Will be very slow for training, but works
- Reduce TIME_BUDGET in prepare.py if just testing

## 4. Key Data Facts

| Fact | Detail |
|------|--------|
| Total train_audio samples | ~35,500 (30K train / 5K val after split) |
| Species count | 234 (162 birds, 35 amphibians, 28 insects, 8 mammals, 1 reptile) |
| Class imbalance | Top species: ~420 samples. Bottom: 1-7 samples |
| Audio format | 32kHz ogg, variable length |
| Prediction window | 5 seconds |
| Test soundscapes | ~600 × 1 min recordings, split into 5-sec windows (~7,200 predictions) |
| Soundscape labels | 1,478 expert-annotated 5-sec segments (valuable — matches test distribution) |
| Rating range | 0-5 (0 = no rating, from iNaturalist) |
| Secondary labels | Some recordings have multiple species — encoded as multi-hot targets |

**Critical**: Some species ONLY appear in `train_soundscapes_labels.csv`, NOT in `train_audio/`. The current baseline doesn't use soundscape labels yet — this is a major improvement opportunity.

## 5. Running the Autoresearch Loop

Read `program.md` for full details. Summary:

```bash
# 1. Create experiment branch
git checkout -b autoresearch/<tag>

# 2. Establish baseline
python train.py > run.log 2>&1
grep "^val_rocauc:\|^peak_memory_mb:" run.log
# Record in results.tsv

# 3. Loop: modify train.py → commit → run → evaluate → keep/discard
# See program.md for the full protocol
```

### The experiment loop (from program.md)

1. Modify `train.py` with an experimental idea
2. `git commit`
3. `python train.py > run.log 2>&1`
4. `grep "^val_rocauc:" run.log` — check result
5. If improved → keep commit. If not → `git reset --hard HEAD~1`
6. Log to `results.tsv`
7. Repeat forever (agent is autonomous, never stops)

### results.tsv format

```
commit	val_rocauc	status	backbone	description
a1b2c3d	0.850000	keep	efficientnet_b0	baseline
```

## 6. What to Experiment With

### High-priority (likely big impact)
- **Fix cosine scheduler**: Current T_max=300 is too small — LR cycles multiple times. Set T_max to match actual total steps (~2600 for 10min budget)
- **Use soundscape labels**: Add `train_soundscapes_labels.csv` data to training — it matches test distribution
- **Focal loss**: Massive class imbalance (420 vs 1 samples). Focal loss down-weights easy examples
- **SpecAugment**: Time and frequency masking on mel spectrograms — proven technique for audio
- **Mixup / CutMix**: Regularization through sample mixing
- **Filter by rating**: Samples with rating ≥ 3 are higher quality

### Medium-priority
- **Backbone swap**: Try `efficientnet_b1`, `convnext_tiny`, `resnet34`, `mobilenetv3_large_100`
- **Attention pooling**: Replace global average pooling with learned attention over time frames
- **Discriminative LR**: Lower LR for pretrained backbone, higher for classification head
- **Label smoothing**: Regularize the BCE loss
- **Longer segments**: Train on 10-sec crops, predict 5-sec windows

### Lower-priority / advanced
- **SED (Sound Event Detection)**: Frame-level predictions with attention pooling
- **Multi-instance learning**: Treat each recording as a bag of 5-sec segments
- **Knowledge distillation**: Train a large model, distill to smaller for CPU inference
- **Test-time augmentation**: Multiple crops + flip at inference
- **Pseudo-labeling**: Use model predictions on unlabeled soundscapes as extra training data

## 7. Inference Constraints (Keep in Mind)

The final Kaggle submission runs on **CPU only**:
- ≤90 minutes for ~600 one-minute soundscapes (~7,200 five-second windows)
- No internet, no GPU
- Model must be bundled in the notebook or loaded from a Kaggle dataset

**Inference budget math**:
| Backbone | CPU speed (segments/sec) | Time for 7,200 segments |
|----------|------------------------|------------------------|
| efficientnet_b0 | ~50 | ~2.5 min |
| efficientnet_b1 | ~35 | ~3.5 min |
| convnext_tiny | ~30 | ~4 min |
| efficientnet_b3 | ~15 | ~8 min |

Plenty of headroom. Even an ensemble of 3-5 models would fit in 90 min.

## 8. Current Baseline Status

- **Backbone**: EfficientNet-B0 (pretrained ImageNet, 1-channel input)
- **Training**: AdamW, LR=1e-3, cosine scheduler (needs T_max fix)
- **Loss**: BCEWithLogitsLoss (no class weighting yet)
- **Data**: train_audio only, 85/15 stratified split, random 5-sec crop
- **Mel spec**: 128 bins, 2048 FFT, 512 hop, 50-14000 Hz
- **Time budget**: 600s (10 min)
- **Known issues**:
  - Cosine scheduler T_max is wrong (cycles too fast)
  - No augmentation
  - No soundscape label data
  - No class balancing for the massive imbalance
  - Baseline val_rocauc: **not yet established** (run was interrupted)

## 9. Quick Start for the Agent

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will:
1. Read `program.md`, `prepare.py`, `train.py`
2. Create an experiment branch
3. Verify data exists (or tell you to download)
4. Run baseline
5. Start the autonomous experiment loop
