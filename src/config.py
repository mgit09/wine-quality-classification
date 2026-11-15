"""Project-wide paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "wine_quality.arff"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

RANDOM_STATE = 42
TARGET_COLUMN = "quality"

# Quality scores up to 5 are low, 6 is medium and 7 or more is high
QUALITY_CLASS_BINS = [-float("inf"), 5, 6, float("inf")]
QUALITY_CLASS_LABELS = ["low", "medium", "high"]
