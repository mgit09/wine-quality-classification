"""Helpers to save figures and tables in the results directory."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator

from src.config import FIGURES_DIR, MODELS_DIR, TABLES_DIR


def save_figure(figure: Figure, file_name: str) -> None:
    """Save a figure in the figures directory and close it.

    Args:
        figure: Figure to be saved.
        file_name: Name of the output file, including the extension.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / file_name, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_table(table: pd.DataFrame, file_name: str) -> None:
    """Save a table as a CSV file in the tables directory.

    Args:
        table: DataFrame to be saved.
        file_name: Name of the output file, including the extension.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / file_name)


def save_model(model: BaseEstimator, file_name: str) -> None:
    """Save a fitted model in the models directory.

    Args:
        model: Fitted model or pipeline.
        file_name: Name of the output file, including the extension.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / file_name)
