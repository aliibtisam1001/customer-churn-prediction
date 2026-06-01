"""
preprocessing.py
----------------
Full preprocessing pipeline:
  - Fix TotalCharges dtype
  - Drop customerID
  - Encode binary + multi-class columns
  - Scale numerics
  - Apply SMOTE for class imbalance
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Columns by type
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
    """
    Full preprocessing pipeline.

    Args:
        df          : Raw DataFrame
        fit_scaler  : True when training, False when predicting single row
        scaler      : Pre-fitted scaler (used when fit_scaler=False)

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = df.copy()

    # --- Fix TotalCharges (stored as string with spaces) ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # --- Drop customer ID ---
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # --- Encode binary columns ---
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP)

    # --- One-hot encode multi-class columns ---
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=False)

    # --- Separate features and target ---
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    feature_names = X.columns.tolist()

    # --- Scale numeric columns ---
    num_cols_present = [c for c in NUMERIC_COLS if c in X.columns]
    if fit_scaler:
        scaler = StandardScaler()
        X[num_cols_present] = scaler.fit_transform(X[num_cols_present])
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    else:
        X[num_cols_present] = scaler.transform(X[num_cols_present])

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- SMOTE on training set only ---
    print(f"[→] Before SMOTE: {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"[✓] After SMOTE:  {pd.Series(y_train).value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def preprocess_single_row(row: dict, scaler, feature_names: list) -> pd.DataFrame:
    """
    Preprocess a single customer dict for live prediction in the Streamlit app.

    Args:
        row          : Dict of raw feature values (from the UI form)
        scaler       : Fitted StandardScaler
        feature_names: Columns the model was trained on

    Returns:
        DataFrame ready for model.predict()
    """
    df = pd.DataFrame([row])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Encode binary
    for col in BINARY_COLS:
        if col in df.columns and col != "Churn":
            df[col] = df[col].map(BINARY_MAP)

    # One-hot encode
    df = pd.get_dummies(df, columns=[c for c in MULTI_COLS if c in df.columns])

    # Align to training feature columns (add missing cols as 0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale numerics
    num_cols_present = [c for c in NUMERIC_COLS if c in df.columns]
    df[num_cols_present] = scaler.transform(df[num_cols_present])

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_raw_data

    df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    print(f"\n[✓] X_train: {X_train.shape}")
    print(f"[✓] X_test:  {X_test.shape}")
    print(f"[✓] Features: {len(features)}")
