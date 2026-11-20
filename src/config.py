"""All paths and settings in one place, so every other file stays short."""
from pathlib import Path

# ---- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "main.csv"                # raw CFPB complaints CSV
MODEL_DIR = ROOT / "models" / "distilbert-complaints"  # fine-tuned model (used by the API)
FIGURES_DIR = ROOT / "reports" / "figures"             # plots from EDA / evaluation

# ---- Data -----------------------------------------------------------------
# Product categories we classify into (CFPB naming used between 2015 and 2017).
PRODUCTS = [
    "bank account or service",
    "credit card",
    "credit reporting",
    "debt collection",
    "money transfers",
    "mortgage",
    "other financial service",
    "payday loan",
    "prepaid card",
    "student loan",
    "virtual currency",
]
YEARS = [2015, 2016, 2017]
MAX_SAMPLES_PER_CLASS = 4000  # cap big classes so they don't dominate training
TEST_SIZE = 0.1
SEED = 32

# Label <-> id mapping (stored inside the saved model, so the API can read it back)
LABEL2ID = {label: i for i, label in enumerate(PRODUCTS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# ---- Model ----------------------------------------------------------------
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 512  # DistilBERT's maximum input length (tokens)
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
