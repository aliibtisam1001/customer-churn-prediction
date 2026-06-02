"""
preprocessing.py
----------------
Full preprocessing pipeline:
  - Fix TotalCharges dtype
  - Drop customerID
  - Encode binary + multi-class columns
  - Scale numerics
  - Handle class imbalance with class_weight (no imbalanced-learn needed)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn"
]
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

MULTI_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def preprocess(df: pd.DataFrame, fit_scaler: bool = True, scaler=None):
    df = df.copy()

    # Fix TotalCharges stored as string
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customer ID
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Encode binary columns
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP)

    # One-hot encode multi-class columns
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=False)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_names = X.columns.tolist()

    # Scale numeric columns
    num_cols_present = [c for c in NUMERIC_COLS if c in X.columns]
    if fit_scaler:
        scaler = StandardScaler()
        X[num_cols_present] = scaler.fit_transform(X[num_cols_present])
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    else:
        X[num_cols_present] = scaler.transform(X[num_cols_present])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle imbalance with oversampling (pure sklearn, no imbalanced-learn)
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    y_train_series = pd.Series(y_train.values, name="Churn")

    X_majority = X_train_df[y_train_series == 0]
    X_minority = X_train_df[y_train_series == 1]
    y_majority = y_train_series[y_train_series == 0]
    y_minority = y_train_series[y_train_series == 1]

    print(f"[→] Before oversampling: {{0: {len(y_majority)}, 1: {len(y_minority)}}}")

    X_minority_upsampled, y_minority_upsampled = resample(
        X_minority, y_minority,
        replace=True,
        n_samples=len(X_majority),
        random_state=42
    )

    X_train_balanced = pd.concat([X_majority, X_minority_upsampled])
    y_train_balanced = pd.concat([y_majority, y_minority_upsampled])

    # Shuffle
    idx = np.random.RandomState(42).permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced.iloc[idx].reset_index(drop=True)
    y_train_balanced = y_train_balanced.iloc[idx].reset_index(drop=True)

    print(f"[✓] After oversampling:  {{0: {len(y_train_balanced[y_train_balanced==0])}, 1: {len(y_train_balanced[y_train_balanced==1])}}}")

    return X_train_balanced, X_test, y_train_balanced, y_test, scaler, feature_names


def preprocess_single_row(row: dict, scaler, feature_names: list) -> pd.DataFrame:
    df = pd.DataFrame([row])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    for col in BINARY_COLS:
        if col in df.columns and col != "Churn":
            df[col] = df[col].map(BINARY_MAP)

    df = pd.get_dummies(df, columns=[c for c in MULTI_COLS if c in df.columns])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    num_cols_present = [c for c in NUMERIC_COLS if c in df.columns]
    df[num_cols_present] = scaler.transform(df[num_cols_present])

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_raw_data
    df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    print(f"[✓] X_train: {X_train.shape}")
    print(f"[✓] X_test:  {X_test.shape}")