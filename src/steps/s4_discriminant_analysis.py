"""Evaluate whether discriminant analysis fits the data and apply LDA and QDA."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import QUALITY_CLASS_LABELS, TRAIN_DATA_PATH
from src.data_loading import load_processed_data, split_features_and_target
from src.evaluation import cross_validate_model
from src.models import (
    create_linear_discriminant_analysis,
    create_quadratic_discriminant_analysis,
)
from src.outputs import save_figure, save_table


def compute_skewness_by_class(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Compute the skewness of every feature inside each class.

    LDA and QDA assume that the features are normally distributed within each
    class, and a skewness far from zero indicates an asymmetric distribution.
    Formal normality tests are not used because, with thousands of rows, they
    reject normality for any real data.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        DataFrame with one row per feature and one column per class.
    """
    skewness = features.groupby(target).skew().T
    return skewness.reindex(columns=QUALITY_CLASS_LABELS)


def compute_standard_deviation_by_class(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Compare the spread of every feature across the classes.

    LDA assumes the same covariance matrix in every class. A ratio close to 1
    between the largest and the smallest standard deviation supports this
    assumption, while large ratios favor QDA.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        DataFrame with the standard deviation per class and the max to min ratio.
    """
    standard_deviations = features.groupby(target).std().T
    standard_deviations = standard_deviations.reindex(columns=QUALITY_CLASS_LABELS)

    # Ratio between the most and the least spread class of each feature
    standard_deviations["max_to_min_ratio"] = standard_deviations.max(
        axis=1
    ) / standard_deviations.min(axis=1)
    return standard_deviations


def find_most_correlated_pairs(
    features: pd.DataFrame,
    number_of_pairs: int = 5,
) -> pd.Series:
    """Find the pairs of features with the strongest absolute correlation.

    Strong correlation between features makes the covariance estimates of the
    discriminant analysis unstable.

    Args:
        features: Feature matrix of the training set.
        number_of_pairs: How many pairs to return.

    Returns:
        Series indexed by feature pairs with the absolute correlation.
    """
    correlation_matrix = features.corr().abs()

    # Keep only the upper triangle to avoid repeated pairs
    upper_triangle_mask = np.triu(
        np.ones(correlation_matrix.shape, dtype=bool),
        k=1,
    )
    pairs = correlation_matrix.where(upper_triangle_mask).stack()

    return pairs.sort_values(ascending=False).head(number_of_pairs)


def compare_discriminant_models(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Compare LDA and QDA with stratified cross-validation.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        DataFrame with the mean and standard deviation of each metric per model.
    """
    models = {
        "LDA": create_linear_discriminant_analysis(),
        "QDA": create_quadratic_discriminant_analysis(),
    }

    results_by_model = {
        model_name: cross_validate_model(model, features, target)
        for model_name, model in models.items()
    }
    return pd.concat(results_by_model, names=["model", "metric"])


def plot_lda_projection(features: pd.DataFrame, target: pd.Series) -> None:
    """Plot the wines projected on the two discriminant axes of the LDA.

    This plot is only used to visualize how well the classes separate, so the
    model is fitted on the whole training set.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.
    """
    model = create_linear_discriminant_analysis()
    projected_features = model.fit_transform(features, target)
    variance_ratios = model.explained_variance_ratio_

    figure, axis = plt.subplots(figsize=(7, 5))

    for label in QUALITY_CLASS_LABELS:
        is_in_class = (target == label).to_numpy()
        axis.scatter(
            projected_features[is_in_class, 0],
            projected_features[is_in_class, 1],
            s=8,
            alpha=0.4,
            label=label,
        )

    axis.set_xlabel(f"Discriminant axis 1 ({variance_ratios[0]:.1%})")
    axis.set_ylabel(f"Discriminant axis 2 ({variance_ratios[1]:.1%})")
    axis.legend(title="Quality class")
    save_figure(figure, "lda_projection.png")


def main() -> None:
    """Check the discriminant analysis assumptions and compare LDA and QDA."""
    train_data = load_processed_data(TRAIN_DATA_PATH)
    features, target = split_features_and_target(train_data)

    # Assumption checks
    skewness = compute_skewness_by_class(features, target)
    standard_deviations = compute_standard_deviation_by_class(features, target)
    correlated_pairs = find_most_correlated_pairs(features)

    print("Skewness by class:")
    print(skewness.round(2))
    print("\nStandard deviation by class:")
    print(standard_deviations.round(3))
    print("\nMost correlated feature pairs:")
    print(correlated_pairs.round(2))

    save_table(skewness, "skewness_by_class.csv")
    save_table(standard_deviations, "standard_deviation_by_class.csv")

    # Cross-validated comparison between the linear and the quadratic version
    comparison = compare_discriminant_models(features, target)
    print("\nCross-validation results:")
    print(comparison.round(3))
    save_table(comparison, "discriminant_analysis_comparison.csv")

    plot_lda_projection(features, target)


if __name__ == "__main__":
    main()
