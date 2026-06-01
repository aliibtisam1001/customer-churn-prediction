"""
train.py
--------
Trains Logistic Regression, Random Forest, and XGBoost.
Saves all models + scaler + feature names to models/.

Usage:
    python src/train.py
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score
)

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_raw_data
from preprocessing import preprocess

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=0.1,
        solver="lbfgs",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),
}


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train all models, print metrics, save to disk."""
    all_results = {}

    for name, model in MODELS.items():
        print(f"\n{'='*50}")
        print(f"  Training: {name}")
        print(f"{'='*50}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        results = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
        }

        # 5-fold CV on AUC
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        results["cv_auc_mean"] = round(cv_scores.mean(), 4)
        results["cv_auc_std"]  = round(cv_scores.std(), 4)

        print(f"  Accuracy  : {results['accuracy']}")
        print(f"  ROC-AUC   : {results['roc_auc']}")
        print(f"  F1 Score  : {results['f1']}")
        print(f"  Precision : {results['precision']}")
        print(f"  Recall    : {results['recall']}")
        print(f"  CV AUC    : {results['cv_auc_mean']} ± {results['cv_auc_std']}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

        all_results[name] = results

        # Save model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe_name}.pkl")
        print(f"  [✓] Saved {safe_name}.pkl")

    # Save results summary
    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Results saved to models/results.json")

    # Save feature names
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    print(f"[✓] Feature names saved")

    return all_results


def load_model(name: str):
    """Load a saved model by name. name = 'logistic_regression' | 'random_forest' | 'xgboost'"""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run: python src/train.py")
    return joblib.load(path)


def load_all_models():
    """Load all 3 trained models. Returns dict."""
    return {
        "Logistic Regression": load_model("logistic_regression"),
        "Random Forest":       load_model("random_forest"),
        "XGBoost":             load_model("xgboost"),
    }


if __name__ == "__main__":
    print("\n🚀 Customer Churn — Model Training Pipeline\n")

    # Load and preprocess
    df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # Train all models
    results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

    print("\n✅ All models trained and saved to models/")
    print("\n--- Final Leaderboard ---")
    for name, r in sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
        print(f"  {name:<25} AUC={r['roc_auc']}  F1={r['f1']}  Recall={r['recall']}")
