"""Compare hold-out, K-fold and leave-one-out to justify the validation choice.

Aula 17 describes the trade-off of each method: hold-out is fast but depends on
one split, leave-one-out uses almost all the data (so its estimate is the least
biased) but is costly and has a high variance, and K-fold sits in between.
This step measures it on the training set, with the same macro F1 for every
method: the estimate itself, how much it changes when the shuffle changes
(variance), how far it is from leave-one-out (bias) and what it costs.

The random forest is left out of leave-one-out, because it would need one fit
of the whole forest per wine (4254 fits), which is the cost argument itself.
The test set is not used.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
)

from src.config import RANDOM_STATE, RESAMPLING_REPEATS, TRAIN_DATA_PATH
from src.data_loading import load_processed_data, split_features_and_target
from src.models import (
    create_logistic_regression,
    create_random_forest,
    create_scaled_linear_discriminant_analysis,
)
from src.outputs import save_table

HOLD_OUT_SHARE = 0.2


def score_predictions(true_labels: pd.Series, predictions: np.ndarray) -> tuple:
    """Score a set of predictions with the macro F1 and the accuracy.

    Args:
        true_labels: Real quality classes.
        predictions: Classes predicted by the model.

    Returns:
        Macro F1 and accuracy.
    """
    return (
        f1_score(true_labels, predictions, average="macro"),
        accuracy_score(true_labels, predictions),
    )


def estimate_with_hold_out(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
) -> tuple:
    """Estimate the performance with one stratified train and validation split.

    Args:
        model: Untrained model or pipeline.
        features: Feature matrix of the training set.
        target: Quality class of each row.
        seed: Seed of the split.

    Returns:
        Macro F1 and accuracy on the validation part.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=HOLD_OUT_SHARE, random_state=seed
    )
    train_index, validation_index = next(splitter.split(features, target))
    fitted = clone(model).fit(features.iloc[train_index], target.iloc[train_index])
    predictions = fitted.predict(features.iloc[validation_index])
    return score_predictions(target.iloc[validation_index], predictions)


def estimate_with_folds(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    splitter,
) -> tuple:
    """Estimate the performance from the out-of-fold predictions of a splitter.

    The predictions of all folds are pooled before scoring, because a single
    wine (leave-one-out) has no F1 of its own.

    Args:
        model: Untrained model or pipeline.
        features: Feature matrix of the training set.
        target: Quality class of each row.
        splitter: Cross-validation splitter.

    Returns:
        Macro F1 and accuracy of the pooled predictions.
    """
    predictions = cross_val_predict(model, features, target, cv=splitter)
    return score_predictions(target, predictions)


def create_methods(features: pd.DataFrame, target: pd.Series) -> dict:
    """Create each validation method as a function of the model and the seed.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        Dictionary from the method name to a tuple with the function, the
        number of fits per run and whether its result depends on the seed.
    """

    def k_fold(number_of_folds: int):
        return lambda model, seed: estimate_with_folds(
            model,
            features,
            target,
            StratifiedKFold(number_of_folds, shuffle=True, random_state=seed),
        )

    return {
        "hold-out 80/20": (
            lambda model, seed: estimate_with_hold_out(model, features, target, seed),
            1,
            True,
        ),
        "5-fold": (k_fold(5), 5, True),
        "10-fold": (k_fold(10), 10, True),
        "leave-one-out": (
            lambda model, seed: estimate_with_folds(
                model, features, target, LeaveOneOut()
            ),
            len(target),
            False,
        ),
    }


def compare_methods(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Compare the validation methods for each model.

    Args:
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        DataFrame with the mean and standard deviation of the macro F1 over
        the repetitions, the gap to leave-one-out, and the cost of each method.
    """
    models = {
        "LDA": create_scaled_linear_discriminant_analysis(use_equal_priors=True),
        "Logistic regression": create_logistic_regression(),
        "Random forest": create_random_forest(),
    }

    rows = {}
    for model_name, model in models.items():
        for method_name, (method, number_of_fits, uses_seed) in create_methods(
            features, target
        ).items():
            if method_name == "leave-one-out" and model_name == "Random forest":
                continue

            seeds = range(RANDOM_STATE, RANDOM_STATE + RESAMPLING_REPEATS)
            start = time.perf_counter()
            estimates = [method(model, seed) for seed in (seeds if uses_seed else [0])]
            seconds = (time.perf_counter() - start) / len(estimates)

            f1_values = [f1 for f1, _ in estimates]
            rows[(model_name, method_name)] = {
                "f1_macro_mean": np.mean(f1_values),
                "f1_macro_std_between_repeats": np.std(f1_values),
                "accuracy_mean": np.mean([accuracy for _, accuracy in estimates]),
                "fits_per_run": number_of_fits,
                "seconds_per_run": seconds,
            }

    comparison = pd.DataFrame(rows).T
    comparison.index.names = ["model", "method"]

    # Leave-one-out is the least biased reference, when it was run
    reference = comparison.xs("leave-one-out", level="method")["f1_macro_mean"]
    comparison["f1_macro_gap_to_leave_one_out"] = [
        comparison.loc[index, "f1_macro_mean"] - reference.get(index[0], np.nan)
        for index in comparison.index
    ]
    return comparison


def main() -> None:
    """Compare the validation methods and save the table."""
    train_data = load_processed_data(TRAIN_DATA_PATH)
    features, target = split_features_and_target(train_data)

    comparison = compare_methods(features, target)
    print(f"Validation methods ({RESAMPLING_REPEATS} repetitions with new shuffles):")
    print(comparison.round(4).to_string())
    save_table(comparison, "resampling_comparison.csv")


if __name__ == "__main__":
    main()
