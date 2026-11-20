"""Metrics and cross-validation helpers shared by the modeling steps."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    CROSS_VALIDATION_FOLDS,
    QUALITY_CLASS_LABELS,
    RANDOM_STATE,
)
from src.outputs import save_figure

CROSS_VALIDATION_SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}


def create_cross_validator() -> StratifiedKFold:
    """Create the stratified k-fold splitter used in every comparison.

    Returns:
        Splitter that keeps the class proportions in each fold.
    """
    return StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def cross_validate_model(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Evaluate a model with stratified cross-validation.

    Args:
        model: Untrained model or pipeline.
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        DataFrame with the mean and standard deviation of each metric.
    """
    fold_scores = cross_validate(
        model,
        features,
        target,
        cv=create_cross_validator(),
        scoring=CROSS_VALIDATION_SCORING,
    )

    summary = {}
    for metric_name in CROSS_VALIDATION_SCORING:
        scores = fold_scores[f"test_{metric_name}"]
        summary[metric_name] = {"mean": scores.mean(), "std": scores.std()}

    return pd.DataFrame(summary).T


def calculate_test_metrics(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
) -> dict[str, float]:
    """Calculate the overall metrics of a set of predictions.

    Precision, recall and F1 use the macro average, which gives the same
    weight to every class regardless of its size.

    Args:
        true_labels: Real quality classes.
        predicted_labels: Classes predicted by the model.

    Returns:
        Dictionary with accuracy, macro precision, macro recall and macro F1.
    """
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision_macro": precision_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
    }


def create_per_class_report(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
) -> pd.DataFrame:
    """Create a table with precision, recall and F1 of each class.

    Args:
        true_labels: Real quality classes.
        predicted_labels: Classes predicted by the model.

    Returns:
        DataFrame with one row per class and the overall averages.
    """
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=QUALITY_CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def plot_confusion_matrix(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    title: str,
    file_name: str,
) -> None:
    """Plot and save the confusion matrix of a set of predictions.

    Args:
        true_labels: Real quality classes.
        predicted_labels: Classes predicted by the model.
        title: Title shown above the matrix.
        file_name: Name of the output image, including the extension.
    """
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=QUALITY_CLASS_LABELS,
    )

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=QUALITY_CLASS_LABELS,
    ).plot(ax=axis, colorbar=False)
    axis.set_title(title)

    save_figure(figure, file_name)
