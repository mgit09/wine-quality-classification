"""Exploratory data analysis of the wine quality dataset."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import (
    CLASS_COLUMN,
    QUALITY_CLASS_LABELS,
    REDUNDANCY_CORRELATION_THRESHOLD,
    TABLES_DIR,
    TARGET_COLUMN,
)
from src.data_loading import (
    create_quality_class,
    load_raw_data,
    remove_duplicate_rows,
)
from src.outputs import save_figure, save_table

HISTOGRAM_BINS = 30
GRID_COLUMNS = 4
SCATTER_MATRIX_FEATURES = 6


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
        axis.set_xticks([])

    figure.tight_layout()
    save_figure(figure, "feature_boxplots.png")


def plot_feature_boxplots_by_class(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Plot the boxplot of every feature for each quality class.

    Features whose distribution shifts between classes help to separate them.

    Args:
        data: DataFrame with the quality class column.
        feature_columns: Names of the feature columns.
    """
    figure, axes = create_subplot_grid(len(feature_columns))

    for axis, column in zip(axes, feature_columns):
        values_by_class = [
            data.loc[data[CLASS_COLUMN] == label, column]
            for label in QUALITY_CLASS_LABELS
        ]
        axis.boxplot(values_by_class)
        axis.set_xticklabels(QUALITY_CLASS_LABELS)
        axis.set_title(column)

    figure.tight_layout()
    save_figure(figure, "feature_boxplots_by_class.png")


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


def calculate_pair_correlations(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """Calculate the absolute correlation of every pair of features.

    Args:
        data: DataFrame with the dataset.
        feature_columns: Names of the feature columns.

    Returns:
        Absolute correlation of each pair, indexed by the two feature names
        and sorted from the largest value down.
    """
    # Keep only the upper triangle so each pair appears once
    correlations = data[feature_columns].corr().abs()
    upper_triangle = np.triu(np.ones(correlations.shape, dtype=bool), k=1)
    pair_correlations = correlations.where(upper_triangle).stack()
    return pair_correlations.dropna().sort_values(ascending=False)


def select_most_correlated_pairs(
    data: pd.DataFrame,
    feature_columns: list[str],
    number_of_pairs: int = 3,
) -> list[tuple[str, str]]:
    """Select the pairs of features with the strongest linear relationship.

    Args:
        data: DataFrame with the dataset.
        feature_columns: Names of the feature columns.
        number_of_pairs: How many pairs to return.

    Returns:
        Pairs of feature names, from the largest absolute correlation down.
    """
    strongest = calculate_pair_correlations(data, feature_columns).head(number_of_pairs)
    print("\nMost correlated feature pairs (absolute correlation):")
    print(strongest.round(3))
    return list(strongest.index)


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


def check_feature_redundancy(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Save and print the pair correlations that may indicate redundancy.

    Args:
        data: DataFrame with the dataset.
        feature_columns: Names of the feature columns.
    """
    pair_correlations = calculate_pair_correlations(data, feature_columns)
    table = pair_correlations.rename("absolute_correlation").to_frame()
    table.index.names = ["feature_1", "feature_2"]
    table["above_threshold"] = (
        table["absolute_correlation"] > REDUNDANCY_CORRELATION_THRESHOLD
    )
    save_table(table, "feature_correlation_pairs.csv")

    number_above = int(table["above_threshold"].sum())
    print("\nAbsolute correlation of every feature pair:")
    print(table["absolute_correlation"].round(3))
    print(
        f"\nPairs above {REDUNDANCY_CORRELATION_THRESHOLD}: {number_above} "
        f"(largest: {table['absolute_correlation'].iloc[0]:.3f})."
    )
    if number_above == 0:
        print("No pair is redundant enough to justify removing a feature.")
    else:
        print("These pairs are candidates for removing one of their features.")


def plot_scatter_matrix(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Plot a scatter matrix of the features colored by quality class.

    The diagonal shows the histogram of each feature per class.

    Args:
        data: DataFrame with the quality class column.
        feature_columns: Names of the features to plot.
    """
    size = len(feature_columns)
    figure, axes = plt.subplots(size, size, figsize=(2.2 * size, 2.2 * size))

    for row, vertical_column in enumerate(feature_columns):
        for column, horizontal_column in enumerate(feature_columns):
            axis = axes[row, column]
            for label in QUALITY_CLASS_LABELS:
                subset = data[data[CLASS_COLUMN] == label]
                if row == column:
                    axis.hist(
                        subset[horizontal_column],
                        bins=HISTOGRAM_BINS,
                        alpha=0.5,
                        label=label,
                    )
                else:
                    axis.scatter(
                        subset[horizontal_column],
                        subset[vertical_column],
                        s=2,
                        alpha=0.3,
                        label=label,
                    )
            axis.tick_params(labelsize=6)
            # Names only on the outer panels, tick labels only on the borders
            if row == size - 1:
                axis.set_xlabel(horizontal_column, fontsize=8)
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(vertical_column, fontsize=8)
            else:
                axis.set_yticklabels([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Quality class",
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    figure.tight_layout()
    save_figure(figure, "scatter_matrix.png")


def main() -> None:
    """Run the exploratory analysis and save its figures and tables."""
    # Same duplicate removal as the preprocessing step, so the EDA describes
    # the wines that the models will use
    data = remove_duplicate_rows(load_raw_data())
    data[CLASS_COLUMN] = create_quality_class(data[TARGET_COLUMN])

    # Every column except the target and the derived class is a feature
    feature_columns = [
        column for column in data.columns if column not in (TARGET_COLUMN, CLASS_COLUMN)
    ]

    save_descriptive_statistics(data)
    plot_class_distribution(data)
    plot_feature_histograms(data, feature_columns)
    plot_feature_boxplots(data, feature_columns)
    plot_feature_boxplots_by_class(data, feature_columns)
    plot_correlation_matrix(data, feature_columns)

    # Scatter plots: the two features most related to quality, then the
    # pairs of features most related to each other (redundancy)
    top_features = select_top_correlated_features(data, feature_columns)
    scatter_pairs = [(top_features[0], top_features[1])]
    for pair in select_most_correlated_pairs(data, feature_columns):
        if pair not in scatter_pairs and pair[::-1] not in scatter_pairs:
            scatter_pairs.append(pair)

    for horizontal_column, vertical_column in scatter_pairs:
        plot_scatter_by_class(data, horizontal_column, vertical_column)

    # Scatter matrix of the features most related to quality
    plot_scatter_matrix(
        data,
        select_top_correlated_features(data, feature_columns, SCATTER_MATRIX_FEATURES),
    )
    check_feature_redundancy(data, feature_columns)


if __name__ == "__main__":
    main()
