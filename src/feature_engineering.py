"""
feature_engineering.py
-----------------------
Encodes the target label, selects top features, scales features,
and writes the final arrays to data/processed/.

Outputs
-------
  data/processed/X_train_scaled.csv
  data/processed/X_test_scaled.csv
  data/processed/y_train_enc.csv
  data/processed/y_test_enc.csv
  models/scaler.pkl
  models/label_encoder.pkl
  models/selected_features.txt
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"

# Number of top features to keep (set to None to keep all)
TOP_N_FEATURES = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_splits(processed_dir: str = PROCESSED_DIR):
    """Load the raw splits produced by preprocessing.py."""
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    print(f"[feature_engineering] Loaded splits — X_train: {X_train.shape}")
    return X_train, X_test, y_train, y_test


def encode_labels(y_train: pd.Series, y_test: pd.Series):
    """
    Encode string labels (M/B) to integers (1/0).
    Returns encoded arrays and the fitted LabelEncoder.
    """
    le = LabelEncoder()
    y_train_enc = pd.Series(le.fit_transform(y_train), name="diagnosis")
    y_test_enc  = pd.Series(le.transform(y_test),      name="diagnosis")
    print(f"[feature_engineering] Label classes: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    return y_train_enc, y_test_enc, le


def select_top_features(
    X_train: pd.DataFrame,
    y_train_enc: pd.Series,
    top_n: int | None = TOP_N_FEATURES,
) -> list[str]:
    """
    Use a quick Random Forest to rank features by importance.
    Returns the list of selected feature names.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_enc)

    importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    importance.sort_values(ascending=False, inplace=True)

    if top_n is not None:
        selected = importance.head(top_n).index.tolist()
        print(f"[feature_engineering] Selected top {top_n} features: {selected}")
    else:
        selected = importance.index.tolist()
        print(f"[feature_engineering] Using all {len(selected)} features (ranked).")

    # Print top-10 summary
    print("\n  Top-10 feature importances:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:<35} {imp:.4f}")
    print()

    return selected


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_features: list[str],
):
    """
    Fit StandardScaler on X_train[selected_features] and transform both splits.
    Returns scaled DataFrames and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[selected_features]),
        columns=selected_features,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[selected_features]),
        columns=selected_features,
    )
    print(f"[feature_engineering] Scaling complete. Shape: {X_train_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler


def save_artifacts(
    X_train_scaled, X_test_scaled,
    y_train_enc, y_test_enc,
    scaler, le, selected_features,
    processed_dir: str = PROCESSED_DIR,
    models_dir: str = MODELS_DIR,
):
    """Persist scaled data, encoders, and feature list."""
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_train_scaled.to_csv(os.path.join(processed_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, "X_test_scaled.csv"),  index=False)
    y_train_enc.to_csv(os.path.join(processed_dir, "y_train_enc.csv"), index=False)
    y_test_enc.to_csv( os.path.join(processed_dir, "y_test_enc.csv"),  index=False)

    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    with open(os.path.join(models_dir, "selected_features.txt"), "w") as f:
        f.write("\n".join(selected_features))

    print(f"[feature_engineering] Artifacts saved to '{processed_dir}/' and '{models_dir}/'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_splits()

    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    selected_features = select_top_features(X_train, y_train_enc)

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, selected_features
    )

    save_artifacts(
        X_train_scaled, X_test_scaled,
        y_train_enc, y_test_enc,
        scaler, le, selected_features,
    )
    print("[feature_engineering] Done.")