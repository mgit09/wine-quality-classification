"""Utilities to load the wine quality dataset and derive the target classes."""

from pathlib import Path

import arff
import pandas as pd

from src.config import (
    CLASS_COLUMN,
    QUALITY_CLASS_BINS,
    QUALITY_CLASS_LABELS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
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


def remove_duplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated rows, keeping the first occurrence.

    This must happen before the train and test split, otherwise the same
    wine could land in both sets and inflate the test metrics. The exploratory
    analysis uses it too, so it describes the same wines the models see.

    Args:
        data: DataFrame with the dataset.

    Returns:
        DataFrame without duplicated rows and with a fresh index.
    """
    number_of_rows_before = len(data)
    cleaned_data = data.drop_duplicates().reset_index(drop=True)

    print(f"Rows before: {number_of_rows_before}")
    print(f"Duplicated rows removed: {number_of_rows_before - len(cleaned_data)}")
    print(f"Rows after: {len(cleaned_data)}")

    return cleaned_data


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


def load_processed_data(file_path: Path) -> pd.DataFrame:
    """Load a processed dataset (train or test) from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with the features, the quality score and the quality class.
    """
    return pd.read_csv(file_path)


def split_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the features from the target class.

    The numeric quality score is dropped from the features because the class
    is derived from it, and keeping it would leak the answer to the models.

    Args:
        data: DataFrame with the features, the score and the class.

    Returns:
        The features DataFrame and the quality class Series.
    """
    features = data.drop(columns=[TARGET_COLUMN, CLASS_COLUMN])
    target = data[CLASS_COLUMN]
    return features, target
