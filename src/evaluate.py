"""
evaluate.py
-----------
Evaluate all trained models and generate:

1. Classification metrics
2. Confusion matrices
3. ROC Curve comparison
4. Evaluation summary CSV
5. Evaluation summary image

Outputs:
---------
outputs/
├── evaluation_summary.csv
├── evaluation_table.png
├── roc_curve_comparison.png
├── confusion_matrix_logistic_regression.png
├── confusion_matrix_svm.png
├── confusion_matrix_random_forest.png
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "SVM": "svm.pkl",
    "Random Forest": "random_forest.pkl"
}


def load_test_data():

    X_test = pd.read_csv(
        os.path.join(
            PROCESSED_DIR,
            "X_test_scaled.csv"
        )
    )

    y_test = pd.read_csv(
        os.path.join(
            PROCESSED_DIR,
            "y_test_enc.csv"
        )
    ).squeeze()

    print(
        f"\nLoaded Test Set: {X_test.shape}"
    )

    return X_test, y_test


def load_models():

    models = {}

    for name, file_name in MODEL_FILES.items():

        path = os.path.join(
            MODELS_DIR,
            file_name
        )

        if os.path.exists(path):

            with open(path, "rb") as f:

                models[name] = pickle.load(f)

        else:

            print(
                f"Model not found: {path}"
            )

    return models


def evaluate_model(
    name,
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X_test)[:, 1]

    else:

        y_prob = None

    acc = accuracy_score(
        y_test,
        y_pred
    )

    prec = precision_score(
        y_test,
        y_pred
    )

    rec = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc = (
        roc_auc_score(
            y_test,
            y_prob
        )
        if y_prob is not None
        else None
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    if roc is not None:
        print(f"ROC AUC   : {roc:.4f}")

    print(
        "\nClassification Report\n"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Benign",
                "Malignant"
            ]
        )
    )

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC AUC": round(roc, 4)
        if roc is not None
        else None
    }


def plot_confusion_matrix(
    name,
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(5, 4))

    plt.imshow(
        cm,
        cmap="Blues"
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Benign", "Malignant"]
    )

    plt.yticks(
        [0, 1],
        ["Benign", "Malignant"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUTS_DIR,
        f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(
        save_path,
        dpi=120
    )

    plt.close()


def plot_roc_curves(
    models,
    X_test,
    y_test
):

    plt.figure(figsize=(8, 6))

    for name, model in models.items():

        if not hasattr(
            model,
            "predict_proba"
        ):
            continue

        probs = model.predict_proba(
            X_test
        )[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probs
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={roc_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUTS_DIR,
            "roc_curve_comparison.png"
        ),
        dpi=120
    )

    plt.close()


if __name__ == "__main__":

    os.makedirs(
        OUTPUTS_DIR,
        exist_ok=True
    )

    X_test, y_test = load_test_data()

    models = load_models()

    results = []

    for name, model in models.items():

        metrics = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results.append(
            metrics
        )

        plot_confusion_matrix(
            name,
            model,
            X_test,
            y_test
        )

    plot_roc_curves(
        models,
        X_test,
        y_test
    )

    summary_df = pd.DataFrame(
        results
    )

    summary_df.sort_values(
        by="ROC AUC",
        ascending=False,
        inplace=True
    )

    summary_path = os.path.join(
        OUTPUTS_DIR,
        "evaluation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nBest Model")
    print("=" * 50)

    print(
        summary_df.iloc[0]
    )

    print(
        "\nEvaluation Completed Successfully."
    )