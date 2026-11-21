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

from src.config import QUALITY_CLASS_LABELS, RANDOM_STATE

# Shrinkage needs the lsqr solver, and None keeps the plain LDA estimate
LINEAR_DISCRIMINANT_ANALYSIS_PARAMETER_GRID = {
    "classifier__solver": ["lsqr"],
    "classifier__shrinkage": [None, "auto", 0.1, 0.3, 0.5],
}

# Keys use the "classifier__" prefix because the model is inside a pipeline
LOGISTIC_REGRESSION_PARAMETER_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
}

RANDOM_FOREST_PARAMETER_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3],
}


def create_linear_discriminant_analysis(
    use_equal_priors: bool = False,
) -> LinearDiscriminantAnalysis:
    """Create a linear discriminant analysis model.

    Args:
        use_equal_priors: If True, every class gets the same prior probability,
            which is how discriminant analysis balances the classes.

    Returns:
        Untrained LDA model. By default it uses the class proportions as priors.
    """
    priors = None
    if use_equal_priors:
        number_of_classes = len(QUALITY_CLASS_LABELS)
        priors = [1 / number_of_classes] * number_of_classes
    return LinearDiscriminantAnalysis(priors=priors)


def create_scaled_linear_discriminant_analysis(
    use_equal_priors: bool = False,
) -> BaseEstimator:
    """Create a linear discriminant analysis with standardized features.

    Scikit-learn only standardizes the features itself for shrinkage="auto".
    A numeric shrinkage pulls the raw covariance toward its average variance,
    which the sulfur dioxide columns dominate, so the scaler is needed to tune
    the shrinkage fairly.

    Args:
        use_equal_priors: If True, every class gets the same prior probability.

    Returns:
        Untrained pipeline with a scaler and an LDA.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                create_linear_discriminant_analysis(use_equal_priors),
            ),
        ]
    )


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


def create_logistic_regression(
    class_weight: str | None = "balanced",
) -> BaseEstimator:
    """Create a logistic regression with standardized features.

    Standardization is part of the pipeline so that, inside cross-validation,
    it is learned only from the training folds.

    Args:
        class_weight: "balanced" to weight each class inversely to its size,
            or None to weight all classes equally.

    Returns:
        Untrained pipeline with a scaler and a logistic regression.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def create_random_forest(class_weight: str | None = "balanced") -> BaseEstimator:
    """Create a random forest.

    No scaling is applied because decision trees are not affected by the
    scale of the features.

    Args:
        class_weight: "balanced" to weight each class inversely to its size,
            or None to weight all classes equally.

    Returns:
        Untrained random forest.
    """
    return RandomForestClassifier(
        class_weight=class_weight,
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
