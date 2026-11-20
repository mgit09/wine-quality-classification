"""Project-wide paths and constants."""

from pathlib import Path

# --------------------------------------------------
# Reproducibility
RANDOM_STATE = 42

# --------------------------------------------------
# Directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

# --------------------------------------------------
# Data files
RAW_DATA_PATH = DATA_DIR / "raw" / "wine_quality.arff"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "wine_quality_train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "wine_quality_test.csv"

# --------------------------------------------------
# Columns
TARGET_COLUMN = "quality"
CLASS_COLUMN = "quality_class"

# --------------------------------------------------
# Quality classes
# Scores up to 5 are low, 6 is medium and 7 or more is high
QUALITY_CLASS_BINS = [-float("inf"), 5, 6, float("inf")]
QUALITY_CLASS_LABELS = ["low", "medium", "high"]

# --------------------------------------------------
# Train and test split and validation
TEST_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5

# --------------------------------------------------
# Hyperparameter tuning
# Macro F1 gives the same weight to every class, including the smaller ones
TUNING_SCORING = "f1_macro"

# --------------------------------------------------
# Final evaluation
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95

# --------------------------------------------------
# PCA
# Share of the variance that the kept components must explain
PCA_VARIANCE_THRESHOLD = 0.90

# --------------------------------------------------
