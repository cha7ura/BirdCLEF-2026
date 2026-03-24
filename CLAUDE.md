# BirdCLEF+ 2026

## Competition
- Kaggle code competition: https://www.kaggle.com/competitions/birdclef-2026
- Goal: Identify 234 wildlife species from audio in the Brazilian Pantanal
- Metric: Macro-averaged ROC-AUC (skipping classes with no true positives)
- CPU-only inference (≤90 min, no internet, no GPU)

## Project Structure (autoresearch pattern)
- `prepare.py` — FIXED. Data loading, dataset, evaluation (ROC-AUC). Do not modify.
- `train.py` — Agent modifies this. Model architecture, optimizer, training loop.
- `program.md` — Agent instructions. Human edits this.
- `requirements.txt` — Dependencies
- `data/` — Competition data (gitignored, 16GB)
- `notebooks/` — EDA notebooks
- `results.tsv` — Experiment log (gitignored, untracked)

## How it works
1. Agent reads `program.md` for instructions
2. Agent modifies `train.py` with an experimental idea
3. Runs `python train.py > run.log 2>&1` (10-minute time budget)
4. Checks result: `grep "^val_rocauc:" run.log`
5. If improved → keep commit. If not → git reset.
6. Log to `results.tsv` and repeat.

## Key Data Facts
- train_audio/: Short species recordings (xeno-canto + iNaturalist), 32kHz ogg
- train_soundscapes/: 1-min field recordings, some with expert labels
- test_soundscapes/: ~600 hidden 1-min recordings (scoring)
- 234 species columns in submission
- Some species ONLY appear in train_soundscapes labels, not train_audio

## Hardware
- Apple M4, 16GB unified memory
- PyTorch MPS backend
- 10-minute training budget per experiment (~6 experiments/hour)
