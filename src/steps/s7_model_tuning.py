"""Tune the hyperparameters of each model with cross-validated grid search.

Every model is tuned on the training set only, with the same stratified folds
and the same scoring, so the comparison is made under equal conditions. Class
balancing stays on, as decided in the balancing step. The best CV score is
optimistic, because the same folds choose the parameters and score them, so the
final performance is measured later on the untouched test set.
"""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from src.config import TRAIN_DATA_PATH, TUNING_SCORING
from src.data_loading import load_processed_data, split_features_and_target
from src.evaluation import create_cross_validator
from src.models import (
    LINEAR_DISCRIMINANT_ANALYSIS_PARAMETER_GRID,
    LOGISTIC_REGRESSION_PARAMETER_GRID,
    RANDOM_FOREST_PARAMETER_GRID,
    create_logistic_regression,
    create_random_forest,
    create_scaled_linear_discriminant_analysis,
)
from src.outputs import save_model, save_table


def create_search_setups() -> dict[str, tuple[BaseEstimator, dict]]:
    """Create each balanced model together with its parameter grid.

    Returns:
        Dictionary from the model name to the model and its grid.
    """
    return {
        "lda": (
            create_scaled_linear_discriminant_analysis(use_equal_priors=True),
            LINEAR_DISCRIMINANT_ANALYSIS_PARAMETER_GRID,
        ),
        "logistic_regression": (
            create_logistic_regression(),
            LOGISTIC_REGRESSION_PARAMETER_GRID,
        ),
        "random_forest": (
            create_random_forest(),
            RANDOM_FOREST_PARAMETER_GRID,
        ),
    }


def tune_model(
    model: BaseEstimator,
    parameter_grid: dict,
    features: pd.DataFrame,
    target: pd.Series,
) -> GridSearchCV:
    """Search the best hyperparameters of a model with cross-validation.

    The search refits the best combination on the whole training set.

    Args:
        model: Untrained model or pipeline.
        parameter_grid: Values to try for each hyperparameter.
        features: Feature matrix of the training set.
        target: Quality class of each row.

    Returns:
        Fitted grid search, which holds the best model and every result.
    """
    search = GridSearchCV(
        model,
        parameter_grid,
        scoring=TUNING_SCORING,
        cv=create_cross_validator(),
        refit=True,
    )
    return search.fit(features, target)


def create_search_results_table(search: GridSearchCV) -> pd.DataFrame:
    """Create a table with the score of every tried combination.

    Args:
        search: Fitted grid search.

    Returns:
        DataFrame sorted from the best to the worst combination.
    """
    results = pd.DataFrame(search.cv_results_)
    columns = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    return results[columns].sort_values("rank_test_score").reset_index(drop=True)


def main() -> None:
    """Tune every model, then save the best ones and the search tables."""
    train_data = load_processed_data(TRAIN_DATA_PATH)
    features, target = split_features_and_target(train_data)

    best_models = {}
    for model_name, (model, parameter_grid) in create_search_setups().items():
        search = tune_model(model, parameter_grid, features, target)

        results_table = create_search_results_table(search)
        print(f"\n{model_name}: {len(results_table)} combinations tried")
        print(results_table.head(3).to_string(index=False))

        save_table(results_table, f"tuning_{model_name}.csv")
        save_model(search.best_estimator_, f"{model_name}.joblib")

        best_models[model_name] = {
            "best_parameters": search.best_params_,
            f"cv_{TUNING_SCORING}": search.best_score_,
        }

    best_table = pd.DataFrame(best_models).T
    print("\nBest parameters:")
    print(best_table.to_string())
    save_table(best_table, "tuning_best_parameters.csv")


if __name__ == "__main__":
    main()
