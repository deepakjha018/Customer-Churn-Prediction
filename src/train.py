"""
Model training utilities.
"""

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)


def get_models(random_state=42):
    """
    Return a dictionary containing all ML models.
    """

    models = {

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "Decision Tree":
        DecisionTreeClassifier(
            random_state=random_state
        ),

        "Random Forest":
        RandomForestClassifier(
            random_state=random_state
        ),

        "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=random_state
        ),

        "Hist Gradient Boosting":
        HistGradientBoostingClassifier(
            random_state=random_state
        )

    }

    return models

def train_models(
    models,
    X_train,
    y_train
):
    """
    Train every model.
    """

    trained_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

    return trained_models