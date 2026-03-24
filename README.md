# BirdCLEF+ 2026

Autonomous experiment pipeline for the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle competition — acoustic species identification in the Pantanal, South America.

Built on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern: an AI agent autonomously modifies `train.py`, runs experiments with a fixed time budget, keeps improvements, discards failures, and repeats.

## Competition

| | |
|---|---|
| **Task** | Identify 234 species (birds, amphibians, mammals, reptiles, insects) from audio |
| **Metric** | Macro-averaged ROC-AUC (skips classes with no true positives) |
| **Data** | ~46K recordings + expert-labeled soundscapes, 32kHz ogg |
| **Submission** | CPU-only Kaggle notebook, ≤90 min, no internet |
| **Prize** | $50,000 |
| **Deadline** | June 3, 2026 |

## Quick Start

```bash
# Clone
git clone https://github.com/cha7ura/BirdCLEF-2026.git
cd BirdCLEF-2026

# Install dependencies
pip install torch torchvision torchaudio timm scikit-learn pandas numpy audiomentations librosa soundfile

# Download data (~16GB) — requires Kaggle CLI configured
kaggle competitions download -c birdclef-2026 -p data/

# Extract and verify
python prepare.py --eda

# Run baseline (~10 min)
python train.py > run.log 2>&1
grep "^val_rocauc:" run.log
```

## How It Works

Three files, inspired by autoresearch:

| File | Role | Who edits |
|------|------|-----------|
| `prepare.py` | Data loading, dataset, evaluation (ROC-AUC) | **Nobody** — fixed |
| `train.py` | Model, optimizer, training loop | **The agent** |
| `program.md` | Experiment loop instructions | **The human** |

The agent reads `program.md`, then loops forever:
1. Modify `train.py` with an idea
2. `git commit`
3. `python train.py > run.log 2>&1` (fixed 10-min budget)
4. Check `val_rocauc` — if improved, keep; otherwise `git reset`
5. Log to `results.tsv`
6. Repeat

## Running the Agent

Start Claude Code in this repo and prompt:

> *"Have a look at program.md and let's kick off a new experiment! Let's do the setup first."*

The agent will create an experiment branch, establish a baseline, and start iterating autonomously. Leave it running overnight for ~50+ experiments.

## Dataset Summary

| | |
|---|---|
| Train samples | 30,295 (+ 5,254 val) |
| Species | 234 (162 birds, 35 amphibians, 28 insects, 8 mammals, 1 reptile) |
| Class imbalance | Top: ~420 samples, Bottom: 1-7 samples |
| Soundscape labels | 1,478 expert-annotated 5-sec segments |
| Audio | 32kHz ogg, 5-second prediction windows |

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| NVIDIA GPU (CUDA) | Recommended | ~80-100ms/step, batch 64+ |
| Apple MPS (M-series) | Works | ~230ms/step, batch 32 |
| CPU | Works (slow) | For testing only |

Auto-detected in `train.py` — no config needed.

## Project Structure

```
├── prepare.py              # Fixed: data, dataset, evaluation
├── train.py                # Agent modifies: model, optimizer, loop
├── program.md              # Agent instructions
├── SETUP.md                # Detailed setup guide for agents
├── CLAUDE.md               # Claude Code project context
├── requirements.txt
├── data/                   # Competition data (gitignored)
│   ├── train_audio/
│   ├── train_soundscapes/
│   ├── train.csv
│   ├── taxonomy.csv
│   └── ...
├── run.log                 # Training output (gitignored)
└── results.tsv             # Experiment log (gitignored)
```

## License

Competition data is [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Code is MIT.
