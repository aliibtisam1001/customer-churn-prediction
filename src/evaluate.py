"""
evaluate.py
-----------
Evaluation utilities: ROC curves, confusion matrix, feature importance plots.
Used by both train.py and the Streamlit app.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------
# ROC Curve (Plotly — used in Streamlit)
# ---------------------------------------------------------------

def plot_roc_curves_plotly(models: dict, X_test, y_test) -> go.Figure:
    """
    Plot ROC curves for all models on one Plotly figure.

    Args:
        models : dict of {model_name: fitted_model}
        X_test, y_test : held-out test data

    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{name} (AUC = {roc_auc:.3f})",
            line=dict(color=color, width=2.5)
        ))

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random (AUC = 0.50)",
        line=dict(color="gray", width=1.5, dash="dash")
    ))

    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------
# Confusion Matrix (Plotly)
# ---------------------------------------------------------------

def plot_confusion_matrix_plotly(model, X_test, y_test, model_name: str) -> go.Figure:
    """Plotly heatmap confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["No Churn", "Churn"]

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels, y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        title=f"Confusion Matrix — {model_name}",
        aspect="auto",
        height=350,
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
    return fig


# ---------------------------------------------------------------
# Feature Importance (Plotly)
# ---------------------------------------------------------------

def plot_feature_importance_plotly(model, feature_names: list, model_name: str, top_n: int = 15) -> go.Figure:
    """
    Bar chart of top-N feature importances.
    Works for Random Forest and XGBoost (tree-based importances).
    For Logistic Regression, uses |coefficients|.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Feature Importances — {model_name}"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = f"|Coefficients| — {model_name}"
    else:
        return go.Figure()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    fig = px.bar(
        df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        color="importance",
        color_continuous_scale="Blues",
        height=450,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=160, r=20, t=50, b=40),
        coloraxis_showscale=False,
    )
    return fig


# ---------------------------------------------------------------
# Churn Distribution Charts (EDA — used in Streamlit)
# ---------------------------------------------------------------

def plot_churn_by_column(df: pd.DataFrame, col: str) -> go.Figure:
    """Grouped bar chart: churn rate per category of `col`."""
    grouped = df.groupby([col, "Churn"]).size().reset_index(name="count")
    fig = px.bar(
        grouped,
        x=col,
        y="count",
        color="Churn",
        barmode="group",
        color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        title=f"Churn by {col}",
        height=380,
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
    return fig


def plot_numeric_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    """Overlapping histogram of a numeric column split by churn."""
    fig = px.histogram(
        df,
        x=col,
        color="Churn",
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        title=f"{col} Distribution by Churn",
        nbins=40,
        height=360,
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
    return fig


def plot_churn_rate_pie(df: pd.DataFrame) -> go.Figure:
    """Simple donut chart of overall churn rate."""
    counts = df["Churn"].value_counts()
    fig = go.Figure(go.Pie(
        labels=["No Churn", "Churn"],
        values=[counts.get("No", 0), counts.get("Yes", 0)],
        hole=0.45,
        marker_colors=["#636EFA", "#EF553B"],
    ))
    fig.update_layout(
        title="Overall Churn Rate",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of numeric feature correlations."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
        height=500,
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=60, r=20, t=50, b=40))
    return fig
