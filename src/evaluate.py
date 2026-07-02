"""
Model evaluation utilities.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model : sklearn estimator

    X_test : Test features

    y_test : Test labels

    Returns
    -------
    metrics : dict
    """

    predictions = model.predict(X_test)

    metrics = {

        "Accuracy":
        accuracy_score(y_test, predictions),

        "Precision":
        precision_score(y_test, predictions),

        "Recall":
        recall_score(y_test, predictions),

        "F1 Score":
        f1_score(y_test, predictions),

        "ROC AUC":
        roc_auc_score(y_test, predictions),

        "Confusion Matrix":
        confusion_matrix(y_test, predictions),

        "Classification Report":
        classification_report(
            y_test,
            predictions
        )

    }

    return metrics


def compare_models(
    trained_models,
    X_test,
    y_test
):
    """
    Compare all trained models.

    Returns
    -------
    pandas.DataFrame
    """

    results = []

    for name, model in trained_models.items():

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results.append({

            "Model": name,

            "Accuracy":
            metrics["Accuracy"],

            "Precision":
            metrics["Precision"],

            "Recall":
            metrics["Recall"],

            "F1 Score":
            metrics["F1 Score"],

            "ROC AUC":
            metrics["ROC AUC"]

        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        by="ROC AUC",
        ascending=False
    )

    return results

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix.
    """

    plt.figure(figsize=(6, 5))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues",
        colorbar=False
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.show()


def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC Curve.
    """

    plt.figure(figsize=(6, 5))

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title("ROC Curve")

    plt.tight_layout()

    plt.show()