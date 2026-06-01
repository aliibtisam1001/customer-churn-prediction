"""
explainability.py
-----------------
SHAP-based model explainability.
  - Global: beeswarm summary plot (which features matter most overall)
  - Local:  waterfall plot (why THIS customer was predicted to churn)
"""

import sys
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def get_shap_explainer(model, X_train: pd.DataFrame, model_name: str):
    """
    Create the right SHAP explainer based on model type.

    - Tree-based (RF, XGBoost): TreeExplainer (fast, exact)
    - Linear (Logistic Reg):    LinearExplainer
    """
    if model_name in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    return explainer


def compute_shap_values(explainer, X: pd.DataFrame):
    """Compute SHAP values for a dataset. Returns shap.Explanation object."""
    shap_values = explainer(X)
    # For binary classifiers, keep only the positive class (churn=1)
    if isinstance(shap_values, list):
        return shap_values[1]
    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]
    return shap_values


def plot_shap_summary(explainer, X_test: pd.DataFrame, model_name: str, max_display: int = 15) -> plt.Figure:
    """
    Global beeswarm summary plot:
    Shows which features push predictions toward churn (red) or away (blue).

    Args:
        explainer   : Fitted SHAP explainer
        X_test      : Test features
        model_name  : For the plot title
        max_display : Number of top features to show

    Returns:
        matplotlib Figure
    """
    shap_values = compute_shap_values(explainer, X_test)

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Summary — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    return fig


def plot_shap_waterfall(explainer, X_row: pd.DataFrame, model_name: str) -> plt.Figure:
    """
    Local waterfall plot for a single customer:
    Shows each feature's contribution to the final churn prediction.

    Args:
        explainer  : Fitted SHAP explainer
        X_row      : Single-row DataFrame (preprocessed)
        model_name : For the plot title

    Returns:
        matplotlib Figure
    """
    shap_values = compute_shap_values(explainer, X_row)

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    plt.title(f"Why this prediction? — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    return fig


def plot_shap_bar(explainer, X_test: pd.DataFrame, model_name: str, top_n: int = 15) -> plt.Figure:
    """
    Global mean |SHAP| bar chart — clean version of feature importance.
    Easier to read than beeswarm for non-technical audiences.
    """
    shap_values = compute_shap_values(explainer, X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.bar(shap_values, max_display=top_n, show=False)
    plt.title(f"Mean |SHAP| Feature Importance — {model_name}", fontsize=12, pad=10)
    plt.tight_layout()
    return fig


def get_top_churn_reasons(explainer, X_row: pd.DataFrame, feature_names: list, top_n: int = 5) -> list:
    """
    Return top-N reasons a customer is predicted to churn as plain text.
    Used in the Streamlit prediction page.

    Returns:
        List of dicts: [{"feature": str, "effect": float, "direction": "increases"/"decreases"}, ...]
    """
    shap_values = compute_shap_values(explainer, X_row)
    sv = shap_values[0].values
    features = feature_names

    reasons = []
    for i, (feat, val) in enumerate(zip(features, sv)):
        reasons.append({
            "feature": feat,
            "shap_value": float(val),
            "direction": "increases churn risk" if val > 0 else "decreases churn risk",
        })

    reasons.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return reasons[:top_n]
