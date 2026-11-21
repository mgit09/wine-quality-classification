"""Clean the raw dataset, derive the quality classes and split train and test."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_COLUMN,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)
from src.data_loading import (
    create_quality_class,
    load_raw_data,
    remove_duplicate_rows,
)


def validate_no_missing_values(data: pd.DataFrame) -> None:
    """Raise an error if the dataset has any missing value.

    No imputation is implemented because the dataset is documented as complete,
    so any missing value means something unexpected happened.

    Args:
        data: DataFrame with the dataset.

    Raises:
        ValueError: If at least one value is missing.
    """
    total_missing_values = int(data.isna().sum().sum())

    if total_missing_values > 0:
        raise ValueError(f"Unexpected missing values found: {total_missing_values}")


def print_class_proportions(name: str, data: pd.DataFrame) -> None:
    """Print the proportion of each quality class in a dataset.

    Args:
        name: Label shown before the proportions.
        data: DataFrame with the quality class column.
    """
    print(f"\nClass proportions in {name}:")
    print(data[CLASS_COLUMN].value_counts(normalize=True).round(3))


def main() -> None:
    """Clean the dataset, split it and save the train and test files."""
    data = load_raw_data()

    # Cleaning steps that do not learn anything from the data
    validate_no_missing_values(data)
    data = remove_duplicate_rows(data)
    data[CLASS_COLUMN] = create_quality_class(data[TARGET_COLUMN])

    # Stratified split keeps the class proportions equal in both sets
    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        stratify=data[CLASS_COLUMN],
        random_state=RANDOM_STATE,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    print(f"\nTrain rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print_class_proportions("train", train_data)
    print_class_proportions("test", test_data)


if __name__ == "__main__":
    main()
