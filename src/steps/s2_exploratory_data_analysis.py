"""Exploratory data analysis of the wine quality dataset."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import (
    FIGURES_DIR,
    QUALITY_CLASS_LABELS,
    TABLES_DIR,
    TARGET_COLUMN,
)
from src.data_loading import create_quality_class, load_raw_data

CLASS_COLUMN = "quality_class"
HISTOGRAM_BINS = 30
GRID_COLUMNS = 4


def create_subplot_grid(number_of_plots: int) -> tuple[Figure, np.ndarray]:
    """Create a grid of axes with room for the requested number of plots.

    Args:
        number_of_plots: How many plots the grid must hold.

    Returns:
        The figure and a flat array with its axes.
    """
    number_of_rows = int(np.ceil(number_of_plots / GRID_COLUMNS))
    figure, axes = plt.subplots(
        number_of_rows,
        GRID_COLUMNS,
        figsize=(4 * GRID_COLUMNS, 3 * number_of_rows),
    )
    axes = axes.flatten()

    # Hide the unused axes of the last row
    for unused_axis in axes[number_of_plots:]:
        unused_axis.axis("off")

    return figure, axes


def save_figure(figure: Figure, file_name: str) -> None:
    """Save a figure in the figures directory and close it.

    Args:
        figure: Figure to be saved.
        file_name: Name of the output file, including the extension.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / file_name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_descriptive_statistics(data: pd.DataFrame) -> None:
    """Save the descriptive statistics of every numeric column as a CSV table.

    Args:
        data: DataFrame with the dataset.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    statistics = data.describe().T
    statistics.to_csv(TABLES_DIR / "descriptive_statistics.csv")
    print(statistics)


def plot_class_distribution(data: pd.DataFrame) -> None:
    """Plot how many wines belong to each quality class.

    Args:
        data: DataFrame with the quality class column.
    """
    class_counts = data[CLASS_COLUMN].value_counts().reindex(QUALITY_CLASS_LABELS)

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(class_counts.index.astype(str), class_counts.values)
    axis.set_xlabel("Quality class")
    axis.set_ylabel("Number of wines")

    # Write the count on top of each bar
    for position, count in enumerate(class_counts.values):
        axis.text(position, count, str(count), ha="center", va="bottom")

    save_figure(figure, "class_distribution.png")


def plot_feature_histograms(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Plot the histogram of every feature.

    Args:
        data: DataFrame with the dataset.
        feature_columns: Names of the feature columns.
    """
    figure, axes = create_subplot_grid(len(feature_columns))

    for axis, column in zip(axes, feature_columns):
        axis.hist(data[column], bins=HISTOGRAM_BINS)
        axis.set_title(column)

    figure.tight_layout()
    save_figure(figure, "feature_histograms.png")


def plot_feature_boxplots(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Plot the boxplot of every feature to reveal outliers.

    Args:
        data: DataFrame with the dataset.
        feature_columns: Names of the feature columns.
    """
    figure, axes = create_subplot_grid(len(feature_columns))

    for axis, column in zip(axes, feature_columns):
        axis.boxplot(data[column])
        axis.set_title(column)

    figure.tight_layout()
    save_figure(figure, "feature_boxplots.png")


def plot_correlation_matrix(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Plot the correlation matrix of the features and the quality score.

    Args:
        data: DataFrame with the dataset.
        feature_columns: Names of the feature columns.
    """
    columns = feature_columns + [TARGET_COLUMN]
    correlation_matrix = data[columns].corr()

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    figure.colorbar(image, ax=axis)

    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels(columns, rotation=90)
    axis.set_yticks(range(len(columns)))
    axis.set_yticklabels(columns)

    # Write the correlation value inside each cell
    for row in range(len(columns)):
        for column in range(len(columns)):
            axis.text(
                column,
                row,
                f"{correlation_matrix.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    save_figure(figure, "correlation_matrix.png")


def select_top_correlated_features(
    data: pd.DataFrame,
    feature_columns: list[str],
    number_of_features: int = 2,
) -> list[str]:
    """Select the features most correlated with the quality score.

    Args:
        data: DataFrame with the dataset.
        feature_columns: Names of the feature columns.
        number_of_features: How many features to return.

    Returns:
        Names of the features with the largest absolute correlation.
    """
    correlations = data[feature_columns].corrwith(data[TARGET_COLUMN])
    print("\nCorrelation with quality:")
    print(correlations.sort_values())

    strongest = correlations.abs().sort_values(ascending=False)
    return strongest.head(number_of_features).index.tolist()


def plot_scatter_by_class(
    data: pd.DataFrame,
    horizontal_column: str,
    vertical_column: str,
) -> None:
    """Plot a scatter plot of two features colored by quality class.

    Args:
        data: DataFrame with the quality class column.
        horizontal_column: Feature shown on the horizontal axis.
        vertical_column: Feature shown on the vertical axis.
    """
    figure, axis = plt.subplots(figsize=(7, 5))

    for label in QUALITY_CLASS_LABELS:
        subset = data[data[CLASS_COLUMN] == label]
        axis.scatter(
            subset[horizontal_column],
            subset[vertical_column],
            s=8,
            alpha=0.4,
            label=label,
        )

    axis.set_xlabel(horizontal_column)
    axis.set_ylabel(vertical_column)
    axis.legend(title="Quality class")
    save_figure(figure, f"scatter_{horizontal_column}_{vertical_column}.png")


def main() -> None:
    """Run the exploratory analysis and save its figures and tables."""
    data = load_raw_data()
    data[CLASS_COLUMN] = create_quality_class(data[TARGET_COLUMN])

    # Every column except the target and the derived class is a feature
    feature_columns = [
        column for column in data.columns if column not in (TARGET_COLUMN, CLASS_COLUMN)
    ]

    save_descriptive_statistics(data)
    plot_class_distribution(data)
    plot_feature_histograms(data, feature_columns)
    plot_feature_boxplots(data, feature_columns)
    plot_correlation_matrix(data, feature_columns)

    # Scatter plot of the two features most related to quality
    top_features = select_top_correlated_features(data, feature_columns)
    plot_scatter_by_class(data, top_features[0], top_features[1])


if __name__ == "__main__":
    main()
