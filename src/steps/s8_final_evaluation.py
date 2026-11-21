"""Evaluate the tuned models once on the held-out test set and compare them.

The test set was never used to choose parameters, so these are the final
estimates. The best model is chosen from the cross-validation score of the
tuning step, and not from the test score, to keep the test set untouched by
any decision. A bootstrap interval shows how much each score could change with
a different sample of wines, which tells whether the differences between the
models are meaningful.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from src.config import (
    BOOTSTRAP_ITERATIONS,
    CONFIDENCE_LEVEL,
    MODELS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TUNING_SCORING,
)
from src.data_loading import load_processed_data, split_features_and_target
from src.evaluation import (
    calculate_auc,
    calculate_test_metrics,
    create_per_class_report,
    plot_confusion_matrix,
    plot_roc_curves,
)
from src.outputs import save_figure, save_table

MODEL_DISPLAY_NAMES = {
    "lda": "LDA",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}


def load_tuned_models() -> dict[str, BaseEstimator]:
    """Load the models saved by the tuning step.

    Returns:
        Dictionary from the model key to the fitted model.

    Raises:
        FileNotFoundError: If the tuning step was not run yet.
    """
    models = {}
    for model_key in MODEL_DISPLAY_NAMES:
        model_path = MODELS_DIR / f"{model_key}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found, run s7_model_tuning first"
            )
        models[model_key] = joblib.load(model_path)
    return models


def bootstrap_macro_f1_interval(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
) -> tuple[float, float]:
    """Estimate a confidence interval of the macro F1 by bootstrap.

    The test wines are resampled with replacement, and the macro F1 of the
    same predictions is computed on each resample.

    Args:
        true_labels: Real quality classes of the test set.
        predicted_labels: Classes predicted by the model.

    Returns:
        Lower and upper limits of the confidence interval.
    """
    random_generator = np.random.default_rng(RANDOM_STATE)
    true_array = true_labels.to_numpy()
    predicted_array = np.asarray(predicted_labels)

    scores = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = random_generator.integers(0, len(true_array), len(true_array))
        scores.append(
            f1_score(
                true_array[sample],
                predicted_array[sample],
                average="macro",
                zero_division=0,
            )
        )

    tail = (1 - CONFIDENCE_LEVEL) / 2
    lower, upper = np.quantile(scores, [tail, 1 - tail])
    return float(lower), float(upper)


def bootstrap_macro_f1_difference(
    true_labels: pd.Series,
    predicted_labels_a: np.ndarray,
    predicted_labels_b: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate the macro F1 difference between two models by paired bootstrap.

    Both models are scored on the same resampled wines, so the difference is
    not blurred by the sample-to-sample variation that the two models share.
    An interval that excludes zero means the first model is really better.

    Args:
        true_labels: Real quality classes of the test set.
        predicted_labels_a: Classes predicted by the first model.
        predicted_labels_b: Classes predicted by the second model.

    Returns:
        Mean difference and the lower and upper limits of its interval.
    """
    random_generator = np.random.default_rng(RANDOM_STATE)
    true_array = true_labels.to_numpy()

    differences = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = random_generator.integers(0, len(true_array), len(true_array))
        scores = [
            f1_score(
                true_array[sample],
                predicted_labels[sample],
                average="macro",
                zero_division=0,
            )
            for predicted_labels in (predicted_labels_a, predicted_labels_b)
        ]
        differences.append(scores[0] - scores[1])

    tail = (1 - CONFIDENCE_LEVEL) / 2
    lower, upper = np.quantile(differences, [tail, 1 - tail])
    return float(np.mean(differences)), float(lower), float(upper)


def compare_models_in_pairs(
    true_labels: pd.Series,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare every pair of real models with the paired bootstrap.

    Args:
        true_labels: Real quality classes of the test set.
        predictions: Predictions of each model, by display name.

    Returns:
        DataFrame with the macro F1 difference of each pair and its interval.
    """
    model_names = [name for name in predictions if name != "Majority class"]

    rows = {}
    for first_index, first_name in enumerate(model_names):
        for second_name in model_names[first_index + 1 :]:
            mean, lower, upper = bootstrap_macro_f1_difference(
                true_labels,
                np.asarray(predictions[first_name]),
                np.asarray(predictions[second_name]),
            )
            rows[f"{first_name} - {second_name}"] = {
                "mean_difference": mean,
                "ci_lower": lower,
                "ci_upper": upper,
            }
    return pd.DataFrame(rows).T


def plot_feature_importances(model: BaseEstimator, feature_names: list[str]) -> None:
    """Plot and save the feature importances of the random forest.

    Args:
        model: Fitted random forest.
        feature_names: Names of the features, in the training order.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values()
    save_table(importances.to_frame("importance"), "random_forest_importances.csv")

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.barh(importances.index, importances.values)
    axis.set_xlabel("Importance")
    save_figure(figure, "random_forest_importances.png")


def plot_importance_comparison(
    model: BaseEstimator,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> None:
    """Compare the impurity and permutation importances of the random forest.

    The impurity importance (MDI) is computed on the training data and favors
    features with many distinct values. The permutation importance is the drop
    of the macro F1 when one feature is shuffled. It is computed on the test
    set, which is acceptable because it only interprets the final model after
    every decision was made and changes none of them. Both are normalized to
    sum to one, so the bars are comparable.

    Args:
        model: Fitted random forest.
        test_features: Feature matrix of the test set.
        test_target: Quality class of each test row.
    """
    result = permutation_importance(
        model,
        test_features,
        test_target,
        scoring=TUNING_SCORING,
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    permutation = pd.DataFrame(
        {
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        },
        index=test_features.columns,
    ).sort_values("importance_mean", ascending=False)
    save_table(permutation, "random_forest_permutation_importances.csv")

    impurity = pd.Series(model.feature_importances_, index=test_features.columns)
    comparison = pd.DataFrame(
        {
            "Impurity (MDI)": impurity / impurity.sum(),
            "Permutation": permutation["importance_mean"]
            / permutation["importance_mean"].sum(),
        }
    ).loc[permutation.index[::-1]]

    figure, axis = plt.subplots(figsize=(8, 6))
    comparison.plot.barh(ax=axis)
    axis.set_xlabel("Normalized importance (sums to 1)")
    save_figure(figure, "random_forest_importance_comparison.png")

    top_impurity = set(impurity.nlargest(3).index)
    top_permutation = set(permutation.index[:3])
    print(f"\nTop 3 by impurity: {sorted(top_impurity)}")
    print(f"Top 3 by permutation: {sorted(top_permutation)}")
    print(f"Top 3 agree: {top_impurity == top_permutation}")


def create_baseline_model(
    train_features: pd.DataFrame,
    train_target: pd.Series,
) -> BaseEstimator:
    """Fit a model that always predicts the most frequent class.

    Any useful model must beat this reference.

    Args:
        train_features: Feature matrix of the training set.
        train_target: Quality class of each training row.

    Returns:
        Fitted baseline model.
    """
    baseline = DummyClassifier(strategy="most_frequent")
    return baseline.fit(train_features, train_target)


def main() -> None:
    """Evaluate every tuned model on the test set and save the comparison."""
    train_features, train_target = split_features_and_target(
        load_processed_data(TRAIN_DATA_PATH)
    )
    test_features, test_target = split_features_and_target(
        load_processed_data(TEST_DATA_PATH)
    )

    models = load_tuned_models()
    named_models = {
        "Majority class": create_baseline_model(train_features, train_target),
        **{MODEL_DISPLAY_NAMES[key]: model for key, model in models.items()},
    }
    predictions = {
        name: model.predict(test_features) for name, model in named_models.items()
    }
    probabilities = {
        name: model.predict_proba(test_features) for name, model in named_models.items()
    }

    # Every model must order its probability columns the same way
    classes = list(named_models["Majority class"].classes_)
    assert all(list(model.classes_) == classes for model in named_models.values())

    rows = {}
    for model_name, predicted in predictions.items():
        metrics = calculate_test_metrics(test_target, predicted)
        metrics.update(calculate_auc(test_target, probabilities[model_name], classes))
        lower, upper = bootstrap_macro_f1_interval(test_target, predicted)
        metrics["f1_macro_ci_lower"] = lower
        metrics["f1_macro_ci_upper"] = upper
        rows[model_name] = metrics

        # Per-class table and confusion matrix of the real models
        if model_name != "Majority class":
            file_key = model_name.lower().replace(" ", "_")
            report = create_per_class_report(test_target, predicted)
            save_table(report, f"test_report_{file_key}.csv")
            plot_confusion_matrix(
                test_target,
                predicted,
                title=model_name,
                file_name=f"confusion_matrix_{file_key}.png",
            )

    comparison = pd.DataFrame(rows).T
    print("Test set results:")
    print(comparison.round(3).to_string())
    save_table(comparison, "test_comparison.csv")

    # A pair differs for real only if its interval excludes zero
    pairs = compare_models_in_pairs(test_target, predictions)
    print("\nPaired bootstrap of the macro F1 difference:")
    print(pairs.round(3).to_string())
    save_table(pairs, "test_paired_differences.csv")

    # The model is chosen from the CV score, so the test set stays untouched
    tuning = pd.read_csv(TABLES_DIR / "tuning_best_parameters.csv", index_col=0)
    best_key = tuning[f"cv_{TUNING_SCORING}"].astype(float).idxmax()
    print(f"\nBest model by cross-validation: {MODEL_DISPLAY_NAMES[best_key]}")

    # ROC curves of the real models, the baseline is the diagonal
    plot_roc_curves(
        test_target,
        {name: p for name, p in probabilities.items() if name != "Majority class"},
        classes,
        "roc_curves.png",
    )

    plot_feature_importances(models["random_forest"], train_features.columns.tolist())
    plot_importance_comparison(models["random_forest"], test_features, test_target)


if __name__ == "__main__":
    main()
