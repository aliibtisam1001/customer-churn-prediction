"""
data_loader.py
--------------
Loads and validates the Telco Customer Churn dataset.
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "telco_churn.csv"


def load_raw_data() -> pd.DataFrame:
    """Load raw CSV and return as DataFrame."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n[!] Dataset not found at {DATA_PATH}\n"
            "    Download it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "    Save as: data/telco_churn.csv"
        )
    df = pd.read_csv(DATA_PATH)
    print(f"[✓] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Basic sanity checks on the raw data."""
    required_cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[!] Missing columns: {missing}")

    print(f"[✓] All required columns present")
    print(f"[✓] Churn distribution:\n{df['Churn'].value_counts()}")
    print(f"[✓] Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return key stats used by the Streamlit app."""
    return {
        "total_customers": len(df),
        "churn_rate": round(df["Churn"].map({"Yes": 1, "No": 0}).mean() * 100, 2),
        "avg_tenure": round(df["tenure"].mean(), 1),
        "avg_monthly_charges": round(df["MonthlyCharges"].mean(), 2),
        "num_features": df.shape[1] - 2,  # minus customerID and Churn
    }


if __name__ == "__main__":
    df = load_raw_data()
    validate_data(df)
    summary = get_data_summary(df)
    print("\n--- Dataset Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
