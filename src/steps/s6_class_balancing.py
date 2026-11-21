"""Measure how class balancing changes the performance of each model.

Cross-validation only estimates performance, it does not change the classes.
Balancing changes how the model is trained, and there are two families:

- Weights or priors: the data stays as it is, and the loss uses class weights
  (logistic regression, random forest) or equal priors (LDA).
- Resampling (Aula 17): the training data itself is changed. Random
  oversampling repeats minority rows (with replacement) until every class has
  the size of the largest one; random undersampling drops majority rows
  (without replacement) until every class has the size of the smallest one.
  These variants use the models without weights or priors, since the
  resampling already balances the classes.

Resampling must happen inside each training fold only. Resampling before the
split would put copies of the same wine in training and validation and inflate
the scores, and the validation fold must keep the real class proportions. So
the resampled variants use their own loop over the same stratified folds,
resample only the training part, and pool the out-of-fold predictions. All
variants are compared under the same folds and metrics.
"""

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample

from src.config import QUALITY_CLASS_LABELS, RANDOM_STATE, TRAIN_DATA_PATH
from src.data_loading import load_processed_data, split_features_and_target
from src.evaluation import (
    calculate_test_metrics,
    create_cross_validator,
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


def resample_training_fold(
    features: pd.DataFrame,
    target: pd.Series,
    oversample: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    """Balance the classes of a training fold by random resampling.

    Args:
        features: Feature matrix of the training fold.
        target: Quality class of each row of the training fold.
        oversample: If True, every class is repeated (with replacement) up to
            the size of the largest class. If False, every class is reduced
            (without replacement) to the size of the smallest class.

    Returns:
        Resampled features and target.
    """
    class_sizes = target.value_counts()
    new_size = class_sizes.max() if oversample else class_sizes.min()

    parts = [
        resample(
            features[target == label],
            replace=oversample,
            n_samples=new_size,
            random_state=RANDOM_STATE,
        )
        for label in class_sizes.index
    ]
    resampled_features = pd.concat(parts)
    return resampled_features, target.loc[resampled_features.index]


def resampled_cross_validated_predictions(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    oversample: bool,
) -> pd.Series:
    """Predict every training row with resampling inside each training fold.

    Args:
        model: Untrained model or pipeline.
        features: Feature matrix of the training set.
        target: Quality class of each row.
        oversample: True for random oversampling, False for undersampling.

    Returns:
        Series with the out-of-fold prediction of each row. The validation
        fold is never resampled.
    """
    predictions = pd.Series(index=target.index, dtype=target.dtype)

    for train_rows, validation_rows in create_cross_validator().split(features, target):
        # Resampling repeats index labels, so it needs a unique index per fold
        fold_features = features.iloc[train_rows].reset_index(drop=True)
        fold_target = target.iloc[train_rows].reset_index(drop=True)
        fold_features, fold_target = resample_training_fold(
            fold_features, fold_target, oversample
        )

        fold_model = clone(model).fit(fold_features, fold_target)
        predictions.iloc[validation_rows] = fold_model.predict(
            features.iloc[validation_rows]
        )

    return predictions


def create_resampling_variants() -> dict[tuple[str, str], tuple[object, bool]]:
    """Create every model without weights or priors, for the resampled data.

    Returns:
        Dictionary indexed by the model name and the balancing setting, with
        the model and whether it uses oversampling (else undersampling).
    """
    models = {
        "LDA": create_linear_discriminant_analysis(),
        "Logistic regression": create_logistic_regression(None),
        "Random forest": create_random_forest(None),
    }
    return {
        (name, balancing): (model, oversample)
        for name, model in models.items()
        for balancing, oversample in [
            ("random oversampling", True),
            ("random undersampling", False),
        ]
    }


def compare_balancing(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Compare every model variant with out-of-fold predictions.

    The overall metrics use the macro average, and the recall of each class
    shows who pays for the balancing. The weight and prior variants come
    first, then the resampled ones.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        DataFrame with one row per model variant.
    """
    rows = {}

    predictions_by_variant = {
        variant: cross_validated_predictions(model, features, target)
        for variant, model in create_model_variants().items()
    }
    for variant, (model, oversample) in create_resampling_variants().items():
        predictions_by_variant[variant] = resampled_cross_validated_predictions(
            model, features, target, oversample
        )

    for variant, predictions in predictions_by_variant.items():
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
