from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ORIGA_DIR = DATA_DIR / "origa"
EYEPACS_DIR = DATA_DIR / "eyepacs"

TRAIN_DIR = ORIGA_DIR / "train"
VAL_DIR   = ORIGA_DIR / "val"
TEST_DIR  = ORIGA_DIR / "test"


EYEPACS_TRAIN_DIR = EYEPACS_DIR / "train"
EYEPACS_VAL_DIR   = EYEPACS_DIR / "validation"
EYEPACS_TEST_DIR  = EYEPACS_DIR / "test"

EYEPACS_METADATA_CSV = EYEPACS_DIR / "metadata.csv"

MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
