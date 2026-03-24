from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_AUDIO_DIR = DATA_DIR / "train_audio"
TRAIN_SOUNDSCAPES_DIR = DATA_DIR / "train_soundscapes"
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_SOUNDSCAPES_LABELS = DATA_DIR / "train_soundscapes_labels.csv"
TAXONOMY_CSV = DATA_DIR / "taxonomy.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"

# Audio
SAMPLE_RATE = 32000
SEGMENT_DURATION = 5  # seconds
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 50
FMAX = 14000

# Training
NUM_CLASSES = 234
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42
