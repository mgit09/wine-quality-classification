"""Print a first overview of the raw wine quality dataset."""

from src.config import TARGET_COLUMN
from src.data_loading import create_quality_class, load_raw_data


def main() -> None:
    """Load the dataset and print its basic characteristics."""
    data = load_raw_data()

    # Shape and column types
    print("Shape:", data.shape)
    print(data.dtypes)

    # Missing values and duplicated rows
    print("\nMissing values per column:")
    print(data.isna().sum())
    print("\nDuplicated rows:", data.duplicated().sum())

    # Descriptive statistics of every column
    print("\nDescriptive statistics:")
    print(data.describe().T)

    # Distribution of the original score and of the grouped classes
    print("\nQuality score counts:")
    print(data[TARGET_COLUMN].value_counts().sort_index())
    print("\nQuality class proportions:")
    print(create_quality_class(data[TARGET_COLUMN]).value_counts(normalize=True))


if __name__ == "__main__":
    main()
