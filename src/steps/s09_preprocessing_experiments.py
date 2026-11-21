"""Test whether removing extreme wines or transforming skewed features helps.

The exploratory analysis found a few extreme points (such as a density of 1.039
and a residual sugar of 65.8 in the same wine) and several right-skewed
features, while LDA and QDA assume normal classes. Both a log transform and a
Yeo-Johnson power transform of the skewed features are tested. Instead of
deciding by eye, each treatment is compared with the untreated data under the same stratified
cross-validation, and only the training set is used, so the test set stays
untouched.

The treatments are applied inside each fold, learned from the training part
only: extreme rows are removed from the training part but the validation part
keeps every wine, so all variants are scored on exactly the same wines. A
treatment is worth adopting only if it improves the macro F1 by more than the
fold-to-fold standard deviation, for most models (this rule is the same for
every treatment, including the power transform).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

from src.config import EXTREME_Z_SCORE, SKEWNESS_THRESHOLD, TRAIN_DATA_PATH
from src.data_loading import load_processed_data, split_features_and_target
from src.evaluation import calculate_test_metrics, create_cross_validator
from src.models import (
    create_logistic_regression,
    create_quadratic_discriminant_analysis,
    create_random_forest,
    create_scaled_linear_discriminant_analysis,
)
from src.outputs import save_table


def find_skewed_features(features: pd.DataFrame) -> list[str]:
    """Find the features whose skewness is above the threshold.

    Args:
        features: Feature matrix of the training set.

    Returns:
        Names of the right-skewed features.
    """
    skewness = features.skew()
    return skewness[skewness > SKEWNESS_THRESHOLD].index.tolist()


def add_log_transform(model: BaseEstimator, columns: list[str]) -> Pipeline:
    """Put a log transform of some columns before a model.

    log1p is used because it is defined for zero, and it is monotonic, so it
    keeps the order of the values and only pulls in the long right tail.

    Args:
        model: Untrained model or pipeline.
        columns: Names of the columns to transform.

    Returns:
        Pipeline that transforms the columns and then runs the model.
    """
    log_transform = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p), columns)],
        remainder="passthrough",
    )
    return Pipeline([("log", log_transform), ("model", model)])


def add_power_transform(model: BaseEstimator, columns: list[str]) -> Pipeline:
    """Put a Yeo-Johnson power transform of some columns before a model.

    Yeo-Johnson is used instead of Box-Cox because Box-Cox needs strictly
    positive values and citric acid has zeros; Yeo-Johnson is defined for any
    value. The exponent of each column is learned from the training part only.
    standardize is False so the output keeps the scale of the data, as in the
    log variant: the models that need scaling already scale it themselves.

    Args:
        model: Untrained model or pipeline.
        columns: Names of the columns to transform.

    Returns:
        Pipeline that transforms the columns and then runs the model.
    """
    power_transform = ColumnTransformer(
        [("power", PowerTransformer(method="yeo-johnson", standardize=False), columns)],
        remainder="passthrough",
    )
    return Pipeline([("power", power_transform), ("model", model)])


def find_non_extreme_rows(features: pd.DataFrame) -> pd.Series:
    """Mark the rows where no feature is extremely far from the mean.

    Args:
        features: Feature matrix of the training part of a fold.

    Returns:
        Boolean Series, True for the rows to keep.
    """
    z_scores = (features - features.mean()) / features.std()
    return (z_scores.abs() <= EXTREME_Z_SCORE).all(axis=1)


def cross_validate_variant(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    remove_extremes: bool,
) -> tuple[dict[str, float], float, int]:
    """Cross-validate a model, optionally dropping extreme training rows.

    Args:
        model: Untrained model or pipeline.
        features: Feature matrix of the training set.
        target: Quality class of each row.
        remove_extremes: Whether to drop extreme rows from each training part.

    Returns:
        Pooled out-of-fold metrics, the standard deviation of the macro F1
        across folds, and how many training rows were dropped in total.
    """
    predictions = pd.Series(index=target.index, dtype=object)
    fold_scores = []
    number_removed = 0

    for train_index, validation_index in create_cross_validator().split(
        features, target
    ):
        train_features = features.iloc[train_index]
        train_target = target.iloc[train_index]

        if remove_extremes:
            keep = find_non_extreme_rows(train_features)
            number_removed += int((~keep).sum())
            train_features, train_target = train_features[keep], train_target[keep]

        fitted = clone(model).fit(train_features, train_target)
        validation_predictions = fitted.predict(features.iloc[validation_index])
        predictions.iloc[validation_index] = validation_predictions
        fold_scores.append(
            f1_score(
                target.iloc[validation_index],
                validation_predictions,
                average="macro",
            )
        )

    metrics = calculate_test_metrics(target, predictions)
    return metrics, float(np.std(fold_scores)), number_removed


def compare_treatments(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Compare every model with each treatment of the data.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        DataFrame with one row per model and treatment, with the change of
        the macro F1 against the untreated data.
    """
    skewed_features = find_skewed_features(features)
    print("Features transformed:", skewed_features)

    models = {
        "LDA": create_scaled_linear_discriminant_analysis(use_equal_priors=True),
        "QDA": create_quadratic_discriminant_analysis(),
        "Logistic regression": create_logistic_regression(),
        "Random forest": create_random_forest(),
    }
    treatments = {
        "none": (False, None),
        "remove extremes": (True, None),
        "log skewed": (False, add_log_transform),
        "both": (True, add_log_transform),
        "power transform": (False, add_power_transform),
    }

    rows = {}
    for model_name, model in models.items():
        for treatment_name, (remove_extremes, add_transform) in treatments.items():
            treated_model = (
                add_transform(model, skewed_features) if add_transform else model
            )
            metrics, f1_std, number_removed = cross_validate_variant(
                treated_model, features, target, remove_extremes
            )
            metrics["f1_macro_fold_std"] = f1_std
            metrics["rows_removed_total"] = number_removed
            rows[(model_name, treatment_name)] = metrics

    comparison = pd.DataFrame(rows).T
    comparison.index.names = ["model", "treatment"]

    baseline = comparison.xs("none", level="treatment")["f1_macro"]
    comparison["f1_macro_change"] = comparison["f1_macro"] - comparison.index.map(
        lambda index: baseline[index[0]]
    )
    return comparison


def main() -> None:
    """Compare the data treatments and save the table."""
    train_data = load_processed_data(TRAIN_DATA_PATH)
    features, target = split_features_and_target(train_data)

    extreme_rows = ~find_non_extreme_rows(features)
    print(f"Training rows beyond {EXTREME_Z_SCORE} standard deviations:")
    print(f"{int(extreme_rows.sum())} of {len(features)}")
    print(target[extreme_rows].value_counts().to_string())

    comparison = compare_treatments(features, target)
    print("\nCross-validated results:")
    print(comparison.round(3).to_string())
    save_table(comparison, "preprocessing_experiments.csv")


if __name__ == "__main__":
    main()
