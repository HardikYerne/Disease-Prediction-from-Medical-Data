import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data():

    df = pd.read_csv("data/data.csv")

    # Remove unnecessary columns
    df.drop(
        columns=["id", "Unnamed: 32"],
        inplace=True
    )

    # Encode target
    encoder = LabelEncoder()

    df["diagnosis"] = encoder.fit_transform(
        df["diagnosis"]
    )

    X = df.drop(
        "diagnosis",
        axis=1
    )

    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    ).to_csv(
        "data/processed/X_train_scaled.csv",
        index=False
    )

    pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    ).to_csv(
        "data/processed/X_test_scaled.csv",
        index=False
    )

    pd.DataFrame(
        y_train
    ).to_csv(
        "data/processed/y_train_enc.csv",
        index=False
    )

    pd.DataFrame(
        y_test
    ).to_csv(
        "data/processed/y_test_enc.csv",
        index=False
    )

    return scaler