import os
import pickle

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def train_models():

    X_train = pd.read_csv(
        "data/processed/X_train_scaled.csv"
    )

    y_train = pd.read_csv(
        "data/processed/y_train_enc.csv"
    ).squeeze()

    os.makedirs(
        "models",
        exist_ok=True
    )

    models = {

        "logistic_regression.pkl":
        LogisticRegression(max_iter=1000),

        "svm.pkl":
        SVC(probability=True),

        "random_forest.pkl":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    }

    for file_name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        with open(
            f"models/{file_name}",
            "wb"
        ) as f:

            pickle.dump(
                model,
                f
            )

        print(
            f"Saved {file_name}"
        )