"""Definition of the classification models and their hyperparameter grids."""

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE

# Keys use the "classifier__" prefix because the model is inside a pipeline
LOGISTIC_REGRESSION_PARAMETER_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
}

RANDOM_FOREST_PARAMETER_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3],
}


def create_linear_discriminant_analysis() -> LinearDiscriminantAnalysis:
    """Create a linear discriminant analysis model.

    Returns:
        Untrained LDA model that uses the class proportions as priors.
    """
    return LinearDiscriminantAnalysis()


def create_quadratic_discriminant_analysis() -> BaseEstimator:
    """Create a quadratic discriminant analysis with standardized features.

    QDA is scale invariant, but scikit-learn discards covariance directions
    with variance below an absolute threshold. Features on small scales, such
    as density, trigger it and the matrix is wrongly reported as singular.
    Standardizing inside the pipeline avoids this without regularization.

    Returns:
        Untrained pipeline with a scaler and a QDA that uses the class
        proportions as priors.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", QuadraticDiscriminantAnalysis()),
        ]
    )


def create_logistic_regression() -> BaseEstimator:
    """Create a logistic regression with standardized features.

    Standardization is part of the pipeline so that, inside cross-validation,
    it is learned only from the training folds.

    Returns:
        Untrained pipeline with a scaler and a balanced logistic regression.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def create_random_forest() -> BaseEstimator:
    """Create a random forest with balanced class weights.

    No scaling is applied because decision trees are not affected by the
    scale of the features.

    Returns:
        Untrained random forest.
    """
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def add_pca(model: BaseEstimator, number_of_components: int) -> Pipeline:
    """Insert a PCA step right before the classifier of a model.

    The features are standardized before the PCA, because the components
    follow the variance and would otherwise be dominated by large-scale
    features such as the sulfur dioxide.

    Args:
        model: Model or pipeline whose last step is the classifier.
        number_of_components: How many principal components to keep.

    Returns:
        Untrained pipeline with the scaler, the PCA and the classifier.
    """
    pca = PCA(n_components=number_of_components, random_state=RANDOM_STATE)

    # Pipelines already have a scaler, so the PCA goes before the classifier
    if isinstance(model, Pipeline):
        steps = list(model.steps)
        steps.insert(len(steps) - 1, ("pca", pca))
        return Pipeline(steps)

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", pca),
            ("classifier", model),
        ]
    )
