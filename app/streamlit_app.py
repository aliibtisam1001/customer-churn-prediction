"""
streamlit_app.py
----------------
5-page interactive Streamlit dashboard:
  1. Home         — project overview + dataset stats
  2. EDA          — interactive exploratory charts
  3. Model Results — ROC curves, confusion matrix, feature importance
  4. Predict      — live customer churn prediction
  5. Explain      — SHAP waterfall for the predicted customer
"""

import subprocess
from pathlib import Path
if not Path("models/xgboost.pkl").exists():
    subprocess.run(["python", "src/train.py"], check=True)
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_loader import load_raw_data, get_data_summary
from preprocessing import preprocess, preprocess_single_row
from evaluate import (
    plot_roc_curves_plotly, plot_confusion_matrix_plotly,
    plot_feature_importance_plotly, plot_churn_by_column,
    plot_numeric_distribution, plot_churn_rate_pie, plot_correlation_heatmap
)
from explainability import (
    get_shap_explainer, plot_shap_summary,
    plot_shap_waterfall, get_top_churn_reasons
)
from train import load_all_models

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Cache expensive operations
# ---------------------------------------------------------------

@st.cache_data
def get_raw_data():
    return load_raw_data()

@st.cache_data
def get_processed_data(_df):
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(_df)
    return X_train, X_test, y_train, y_test, scaler, feature_names

@st.cache_resource
def get_models():
    return load_all_models()

@st.cache_resource
def get_results():
    p = ROOT / "models" / "results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

@st.cache_resource
def get_explainer(_model, _X_train, model_name):
    return get_shap_explainer(_model, _X_train, model_name)

# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title("📉 Churn Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 EDA", "🤖 Model Results", "🔮 Predict", "🔍 Explain"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Telco Customer Churn")
st.sidebar.markdown("**Models:** LR · RF · XGBoost")
st.sidebar.markdown("[GitHub](https://github.com) · [Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
try:
    df_raw = get_raw_data()
    models_loaded = True
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

try:
    X_train, X_test, y_train, y_test, scaler, feature_names = get_processed_data(df_raw)
    models = get_models()
    results = get_results()
except FileNotFoundError:
    st.warning("⚠️ Models not trained yet. Run: `python src/train.py`")
    models = None
    results = {}

# ---------------------------------------------------------------
# Page 1 — Home
# ---------------------------------------------------------------
if page == "🏠 Home":
    st.title("📉 Customer Churn Prediction")
    st.markdown(
        "A production-style ML pipeline that predicts which telecom customers will churn "
        "— and **explains exactly why** using SHAP values."
    )
    st.markdown("---")

    summary = get_data_summary(df_raw)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{summary['total_customers']:,}")
    col2.metric("Churn Rate", f"{summary['churn_rate']}%")
    col3.metric("Avg Tenure", f"{summary['avg_tenure']} months")
    col4.metric("Avg Monthly Charges", f"${summary['avg_monthly_charges']}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🗂️ Pipeline Overview")
        st.markdown("""
1. **Load & Validate** — 7,043 rows, 21 columns
2. **Preprocess** — fix types, encode, scale, SMOTE
3. **Train** — Logistic Regression, Random Forest, XGBoost
4. **Evaluate** — ROC-AUC, F1, Precision, Recall
5. **Explain** — SHAP global + local explanations
6. **Deploy** — This Streamlit app
        """)

    with col_b:
        st.subheader("📁 Dataset Features")
        st.dataframe(
            df_raw.drop(columns=["customerID"]).head(8),
            use_container_width=True,
            height=260
        )

    if results:
        st.markdown("---")
        st.subheader("🏆 Model Leaderboard")
        leaderboard = pd.DataFrame(results).T.reset_index()
        leaderboard.columns = ["Model", "Accuracy", "ROC-AUC", "F1", "Precision", "Recall", "CV AUC Mean", "CV AUC Std"]
        leaderboard = leaderboard.sort_values("ROC-AUC", ascending=False)
        st.dataframe(leaderboard.set_index("Model").style.highlight_max(axis=0, color="#d4f1d4"), use_container_width=True)

# ---------------------------------------------------------------
# Page 2 — EDA
# ---------------------------------------------------------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Explore how churn relates to customer attributes.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_churn_rate_pie(df_raw), use_container_width=True)
    with col2:
        st.plotly_chart(plot_churn_by_column(df_raw, "Contract"), use_container_width=True)

    st.markdown("---")
    cat_col = st.selectbox(
        "Churn by categorical feature:",
        ["InternetService", "PaymentMethod", "gender", "Partner", "Dependents", "SeniorCitizen"]
    )
    st.plotly_chart(plot_churn_by_column(df_raw, cat_col), use_container_width=True)

    st.markdown("---")
    num_col = st.selectbox(
        "Distribution of numeric feature:",
        ["tenure", "MonthlyCharges", "TotalCharges"]
    )
    st.plotly_chart(plot_numeric_distribution(df_raw, num_col), use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    df_encoded = df_raw.copy()
    df_encoded["Churn_binary"] = df_encoded["Churn"].map({"Yes": 1, "No": 0})
    df_encoded["TotalCharges"] = pd.to_numeric(df_encoded["TotalCharges"], errors="coerce")
    st.plotly_chart(plot_correlation_heatmap(df_encoded), use_container_width=True)

# ---------------------------------------------------------------
# Page 3 — Model Results
# ---------------------------------------------------------------
elif page == "🤖 Model Results":
    st.title("🤖 Model Results")

    if not models:
        st.warning("Run `python src/train.py` first.")
        st.stop()

    st.markdown("---")
    st.subheader("ROC Curves")
    st.plotly_chart(plot_roc_curves_plotly(models, X_test, y_test), use_container_width=True)

    st.markdown("---")
    selected_model_name = st.selectbox("Select model for detailed analysis:", list(models.keys()))
    selected_model = models[selected_model_name]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_confusion_matrix_plotly(selected_model, X_test, y_test, selected_model_name),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            plot_feature_importance_plotly(selected_model, feature_names, selected_model_name),
            use_container_width=True
        )

    if results:
        st.markdown("---")
        st.subheader("Full Metrics Table")
        r = results.get(selected_model_name, {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  r.get("accuracy", "—"))
        c2.metric("ROC-AUC",   r.get("roc_auc", "—"))
        c3.metric("F1 Score",  r.get("f1", "—"))
        c4.metric("Precision", r.get("precision", "—"))
        c5.metric("Recall",    r.get("recall", "—"))

# ---------------------------------------------------------------
# Page 4 — Predict
# ---------------------------------------------------------------
elif page == "🔮 Predict":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Fill in a customer's details and get an instant churn probability.")

    if not models:
        st.warning("Run `python src/train.py` first.")
        st.stop()

    st.markdown("---")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender          = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen  = st.selectbox("Senior Citizen", [0, 1])
            partner         = st.selectbox("Partner", ["Yes", "No"])
            dependents      = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            st.markdown("**Services**")
            tenure              = st.slider("Tenure (months)", 0, 72, 12)
            phone_service       = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines      = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service    = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security     = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup       = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection   = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support        = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv        = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies    = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("**Billing**")
            contract            = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing   = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method      = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges     = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total_charges       = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                                   value=float(monthly_charges * tenure))

        model_choice = st.selectbox("Model", list(models.keys()), index=2)
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        raw_row = {
            "gender": gender, "SeniorCitizen": senior_citizen,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }

        X_row = preprocess_single_row(raw_row, scaler, feature_names)
        model = models[model_choice]
        prob = model.predict_proba(X_row)[0][1]
        pred = "Churn" if prob >= 0.5 else "No Churn"

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Prediction", pred)
        col_r2.metric("Churn Probability", f"{prob*100:.1f}%")
        risk = "🔴 High" if prob >= 0.7 else ("🟡 Medium" if prob >= 0.4 else "🟢 Low")
        col_r3.metric("Risk Level", risk)

        if prob >= 0.7:
            st.error("⚠️ This customer is at high risk of churning. Intervention recommended.")
        elif prob >= 0.4:
            st.warning("🟡 Moderate churn risk. Consider a retention offer.")
        else:
            st.success("✅ This customer is likely to stay.")

        # Store for Explain page
        st.session_state["last_X_row"] = X_row
        st.session_state["last_model_name"] = model_choice
        st.session_state["last_prob"] = prob
        st.info("👉 Go to **Explain** page to see why this prediction was made.")

# ---------------------------------------------------------------
# Page 5 — Explain
# ---------------------------------------------------------------
elif page == "🔍 Explain":
    st.title("🔍 SHAP Explainability")
    st.markdown("Understand **why** the model makes its predictions — globally and per customer.")

    if not models:
        st.warning("Run `python src/train.py` first.")
        st.stop()

    st.markdown("---")
    tab1, tab2 = st.tabs(["🌍 Global Explanation", "👤 Customer Explanation"])

    with tab1:
        st.subheader("Global Feature Impact (SHAP Summary)")
        st.markdown(
            "Each dot is one customer. **Red** = high feature value, **Blue** = low. "
            "Position on x-axis = impact on churn prediction."
        )
        model_name_g = st.selectbox("Model:", list(models.keys()), key="global_model")
        model_g = models[model_name_g]

        with st.spinner("Computing SHAP values (may take 20–30s) ..."):
            # Use a sample of 300 rows for speed
            X_sample = X_test.sample(min(300, len(X_test)), random_state=42)
            explainer_g = get_explainer(model_g, X_train, model_name_g)
            fig_summary = plot_shap_summary(explainer_g, X_sample, model_name_g)
            st.pyplot(fig_summary, use_container_width=True)

    with tab2:
        st.subheader("Why did the model predict this customer's outcome?")

        if "last_X_row" not in st.session_state:
            st.info("Go to the **Predict** page first, make a prediction, then come back here.")
        else:
            X_row        = st.session_state["last_X_row"]
            model_name_l = st.session_state["last_model_name"]
            prob         = st.session_state["last_prob"]
            model_l      = models[model_name_l]

            st.markdown(f"**Model:** {model_name_l} · **Churn Probability:** {prob*100:.1f}%")

            with st.spinner("Computing local SHAP values ..."):
                explainer_l = get_explainer(model_l, X_train, model_name_l)
                fig_wf = plot_shap_waterfall(explainer_l, X_row, model_name_l)
                st.pyplot(fig_wf, use_container_width=True)

            st.markdown("---")
            st.subheader("Top reasons in plain English")
            reasons = get_top_churn_reasons(explainer_l, X_row, feature_names, top_n=5)
            for i, r in enumerate(reasons, 1):
                direction_icon = "⬆️" if r["direction"].startswith("increases") else "⬇️"
                st.markdown(
                    f"**{i}.** `{r['feature']}` — "
                    f"{direction_icon} {r['direction']} "
                    f"(SHAP = {r['shap_value']:+.4f})"
                )
