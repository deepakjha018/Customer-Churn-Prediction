"""
Pipeline utilities.
"""

from sklearn.pipeline import Pipeline


def create_pipeline(preprocessor, model):
    """
    Create a complete machine learning pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Feature preprocessing pipeline.

    model : sklearn estimator
        Trained machine learning model.

    Returns
    -------
    Pipeline
    """

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline