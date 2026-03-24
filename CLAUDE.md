# BirdCLEF+ 2026

## Competition
- Kaggle code competition: https://www.kaggle.com/competitions/birdclef-2026
- Goal: Identify 234 wildlife species from audio in the Brazilian Pantanal
- Metric: Macro-averaged ROC-AUC (skipping classes with no true positives)
- CPU-only inference (≤90 min, no internet, no GPU)

## Project Structure
- `data/` — raw competition data (gitignored)
- `notebooks/` — EDA and experiment notebooks
- `src/` — reusable modules (dataset, models, training, inference)
- `configs/` — experiment configs
- `submissions/` — generated submission CSVs

## Tech Stack
- Python 3.10+
- PyTorch + torchaudio
- librosa for audio processing
- timm for pretrained models (e.g., EfficientNet, ConvNeXt)
- Audio: 32kHz ogg format, 5-second prediction windows

## Key Data Facts
- train_audio/: Short species recordings (xeno-canto + iNaturalist)
- train_soundscapes/: 1-min field recordings, some with expert labels
- test_soundscapes/: ~600 hidden 1-min recordings (scoring)
- 234 species columns in submission (birds, amphibians, mammals, reptiles, insects)
- Some species ONLY appear in train_soundscapes labels, not train_audio
