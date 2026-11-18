"""Utilities to load the wine quality dataset and derive the target classes."""

from pathlib import Path

import arff
import pandas as pd

from src.config import (
    QUALITY_CLASS_BINS,
    QUALITY_CLASS_LABELS,
    RAW_DATA_PATH,
)


def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw wine quality dataset from an ARFF file.

    Args:
        file_path: Path to the ARFF file.

    Returns:
        DataFrame with the physicochemical features and the quality score.
    """
    with file_path.open(encoding="utf-8") as arff_file:
        arff_content = arff.load(arff_file)

    # Each attribute is a (name, type) pair
    column_names = [name for name, _ in arff_content["attributes"]]

    return pd.DataFrame(
        arff_content["data"],
        columns=column_names,
    )


def create_quality_class(quality_scores: pd.Series) -> pd.Series:
    """Group the numeric quality score into ordered classes.

    Args:
        quality_scores: Series with the quality score of each wine.

    Returns:
        Ordered categorical Series with the labels low, medium and high.
    """
    return pd.cut(
        quality_scores,
        bins=QUALITY_CLASS_BINS,
        labels=QUALITY_CLASS_LABELS,
        ordered=True,
    )
