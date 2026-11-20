"""Measure how class balancing changes the performance of each model.

Cross-validation only estimates performance, it does not change the classes.
Balancing changes how the model is trained: class weights in the loss, or equal
priors in the discriminant analysis. This step compares both settings under
the same cross-validation, so the effect of balancing is isolated.
"""

import pandas as pd

from src.config import QUALITY_CLASS_LABELS, TRAIN_DATA_PATH
from src.data_loading import load_processed_data, split_features_and_target
from src.evaluation import (
    calculate_test_metrics,
    create_per_class_report,
    cross_validated_predictions,
)
from src.models import (
    create_linear_discriminant_analysis,
    create_logistic_regression,
    create_random_forest,
)
from src.outputs import save_table


def create_model_variants() -> dict[tuple[str, str], object]:
    """Create every model with and without class balancing.

    Returns:
        Dictionary indexed by the model name and the balancing setting.
    """
    return {
        ("LDA", "class proportions as priors"): create_linear_discriminant_analysis(),
        ("LDA", "equal priors"): create_linear_discriminant_analysis(
            use_equal_priors=True
        ),
        ("Logistic regression", "no balancing"): create_logistic_regression(None),
        ("Logistic regression", "balanced weights"): create_logistic_regression(),
        ("Random forest", "no balancing"): create_random_forest(None),
        ("Random forest", "balanced weights"): create_random_forest(),
    }


def compare_balancing(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Compare every model variant with out-of-fold predictions.

    The overall metrics use the macro average, and the recall of each class
    shows who pays for the balancing.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        DataFrame with one row per model variant.
    """
    rows = {}

    for variant, model in create_model_variants().items():
        predictions = cross_validated_predictions(model, features, target)
        metrics = calculate_test_metrics(target, predictions)

        report = create_per_class_report(target, predictions)
        for label in QUALITY_CLASS_LABELS:
            metrics[f"recall_{label}"] = report.loc[label, "recall"]

        rows[variant] = metrics

    comparison = pd.DataFrame(rows).T
    comparison.index.names = ["model", "balancing"]
    return comparison


def main() -> None:
    """Compare the models with and without class balancing."""
    train_data = load_processed_data(TRAIN_DATA_PATH)
    features, target = split_features_and_target(train_data)

    print("Class proportions in the training set:")
    print(target.value_counts(normalize=True).round(3))

    comparison = compare_balancing(features, target)
    print("\nCross-validated results:")
    print(comparison.round(3))
    save_table(comparison, "class_balancing_comparison.csv")


if __name__ == "__main__":
    main()
