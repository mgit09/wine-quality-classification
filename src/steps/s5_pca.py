"""Evaluate whether PCA is useful and measure its effect on the models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    PCA_VARIANCE_THRESHOLD,
    QUALITY_CLASS_LABELS,
    RANDOM_STATE,
    TRAIN_DATA_PATH,
)
from src.data_loading import load_processed_data, split_features_and_target
from src.evaluation import cross_validate_model
from src.models import (
    add_pca,
    create_linear_discriminant_analysis,
    create_logistic_regression,
    create_random_forest,
)
from src.outputs import save_figure, save_table


def fit_full_pca(features: pd.DataFrame) -> PCA:
    """Fit a PCA with every component on the standardized features.

    The features are standardized because they have very different scales,
    and the PCA would otherwise follow only the largest ones.

    Args:
        features: Feature matrix of the training set.

    Returns:
        PCA fitted with as many components as there are features.
    """
    standardized_features = StandardScaler().fit_transform(features)
    return PCA(random_state=RANDOM_STATE).fit(standardized_features)


def create_variance_table(pca: PCA) -> pd.DataFrame:
    """Create a table with the variance explained by each component.

    Args:
        pca: PCA fitted with every component.

    Returns:
        DataFrame with the eigenvalue, the explained variance and its
        cumulative sum for each component.
    """
    component_names = [f"PC{index + 1}" for index in range(pca.n_components_)]
    return pd.DataFrame(
        {
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        },
        index=component_names,
    )


def choose_number_of_components(variance_table: pd.DataFrame) -> int:
    """Find the fewest components that reach the variance threshold.

    Args:
        variance_table: Table created by create_variance_table.

    Returns:
        Number of components whose cumulative variance reaches the threshold.
    """
    reached_threshold = variance_table["cumulative_variance_ratio"].to_numpy()
    return int(np.argmax(reached_threshold >= PCA_VARIANCE_THRESHOLD)) + 1


def count_kaiser_components(variance_table: pd.DataFrame) -> int:
    """Count the components with an eigenvalue above 1 (Kaiser criterion).

    On standardized features, a component with an eigenvalue below 1 explains
    less variance than a single original feature.

    Args:
        variance_table: Table created by create_variance_table.

    Returns:
        Number of components with an eigenvalue above 1.
    """
    return int((variance_table["eigenvalue"] > 1).sum())


def plot_explained_variance(
    variance_table: pd.DataFrame,
    number_of_components: int,
) -> None:
    """Plot the scree plot and the cumulative explained variance.

    Args:
        variance_table: Table created by create_variance_table.
        number_of_components: Number of components chosen for the models.
    """
    positions = np.arange(1, len(variance_table) + 1)
    figure, (scree_axis, cumulative_axis) = plt.subplots(1, 2, figsize=(11, 4))

    scree_axis.bar(positions, variance_table["explained_variance_ratio"])
    scree_axis.set_xlabel("Principal component")
    scree_axis.set_ylabel("Explained variance ratio")
    scree_axis.set_xticks(positions)

    cumulative_axis.plot(
        positions,
        variance_table["cumulative_variance_ratio"],
        marker="o",
    )
    cumulative_axis.axhline(
        PCA_VARIANCE_THRESHOLD,
        color="gray",
        linestyle="--",
        label=f"{PCA_VARIANCE_THRESHOLD:.0%} threshold",
    )
    cumulative_axis.axvline(
        number_of_components,
        color="red",
        linestyle=":",
        label=f"{number_of_components} components",
    )
    cumulative_axis.set_xlabel("Number of components")
    cumulative_axis.set_ylabel("Cumulative explained variance")
    cumulative_axis.set_xticks(positions)
    cumulative_axis.legend()

    figure.tight_layout()
    save_figure(figure, "pca_explained_variance.png")


def create_loadings_table(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """Create a table with the weight of each feature in the first components.

    Args:
        pca: PCA fitted with every component.
        feature_names: Names of the original features.

    Returns:
        DataFrame with one row per feature and one column per component.
    """
    component_names = [f"PC{index + 1}" for index in range(pca.n_components_)]
    return pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=component_names,
    )


def plot_pca_projection(
    features: pd.DataFrame,
    target: pd.Series,
    pca: PCA,
) -> None:
    """Plot the wines projected on the first two principal components.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.
        pca: PCA fitted with every component.
    """
    standardized_features = StandardScaler().fit_transform(features)
    projected_features = pca.transform(standardized_features)
    variance_ratios = pca.explained_variance_ratio_

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

    axis.set_xlabel(f"PC1 ({variance_ratios[0]:.1%})")
    axis.set_ylabel(f"PC2 ({variance_ratios[1]:.1%})")
    axis.legend(title="Quality class")
    save_figure(figure, "pca_projection.png")


def compare_models_with_and_without_pca(
    features: pd.DataFrame,
    target: pd.Series,
    number_of_components: int,
) -> pd.DataFrame:
    """Compare each model with and without PCA using cross-validation.

    The scaler and the PCA are inside the pipeline, so inside each fold they
    are learned only from the training part.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.
        number_of_components: How many principal components to keep.

    Returns:
        DataFrame with the mean and standard deviation of each metric for
        every model and preprocessing combination.
    """
    base_models = {
        "LDA": create_linear_discriminant_analysis(),
        "Logistic regression": create_logistic_regression(),
        "Random forest": create_random_forest(),
    }

    results = {}
    for model_name, model in base_models.items():
        results[(model_name, "original features")] = cross_validate_model(
            model, features, target
        )
        results[(model_name, f"PCA {number_of_components} components")] = (
            cross_validate_model(
                add_pca(model, number_of_components),
                features,
                target,
            )
        )

    return pd.concat(results, names=["model", "features", "metric"])


def main() -> None:
    """Analyze the principal components and compare the models with PCA."""
    train_data = load_processed_data(TRAIN_DATA_PATH)
    features, target = split_features_and_target(train_data)

    pca = fit_full_pca(features)
    variance_table = create_variance_table(pca)
    number_of_components = choose_number_of_components(variance_table)

    print("Explained variance:")
    print(variance_table.round(3))
    print(
        f"\nComponents to reach {PCA_VARIANCE_THRESHOLD:.0%} of the variance:",
        number_of_components,
    )
    print(
        "Components with eigenvalue above 1:", count_kaiser_components(variance_table)
    )

    loadings = create_loadings_table(pca, features.columns.tolist())
    print("\nLoadings of the first components:")
    print(loadings.iloc[:, :number_of_components].round(2))

    save_table(variance_table, "pca_explained_variance.csv")
    save_table(loadings, "pca_loadings.csv")
    plot_explained_variance(variance_table, number_of_components)
    plot_pca_projection(features, target, pca)

    # Effect of the dimensionality reduction on the models
    comparison = compare_models_with_and_without_pca(
        features,
        target,
        number_of_components,
    )
    print("\nCross-validation results:")
    print(comparison.round(3))
    save_table(comparison, "pca_model_comparison.csv")


if __name__ == "__main__":
    main()
