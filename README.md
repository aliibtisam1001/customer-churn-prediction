# 📉 Telecom Customer Churn Prediction Dashboard

A production-grade machine learning project that predicts customer churn risk and delivers model transparency using **SHAP (SHapley Additive exPlanations)** values. The model comparisons and explanations are presented via an interactive, custom-styled 5-page Streamlit dashboard.

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit&logoColor=white&style=flat-square)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-blue?logo=scikit-learn&logoColor=white&style=flat-square)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen?style=flat-square)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🚀 Interactive App Demo

The application can be deployed directly to [Streamlit Community Cloud](https://streamlit.io/cloud) or [Hugging Face Spaces](https://huggingface.co/spaces). 

---

## 📌 Features & Key Capabilities

This project implements an end-to-end data science pipeline, from exploratory analysis to explainable model deployment:

*   **Interactive Exploratory Data Analysis (EDA):** Visualizes churn distributions, categorical factors (contracts, payment methods, internet service), numeric characteristics (tenure, monthly charges), and feature correlations using Plotly.
*   **Robust Data Preprocessing:** Includes data type alignment, missing value imputation, binary/one-hot encoding, feature scaling, and class-imbalance correction via stratified minority oversampling.
*   **Multi-Model Training & Benchmarking:** Evaluates and compares three distinct classifiers (Logistic Regression, Random Forest, and XGBoost) using cross-validation and test set metrics.
*   **Global & Local Explainability:** Uses SHAP explainer objects to extract global feature impacts (beeswarm plot, mean SHAP importance) and customer-specific risk drivers (waterfall plot + plain English reasons).
*   **Premium Web Application:** Implements a multi-page dashboard featuring a dark sidebar navigation, linear-gradient hero panels, KPI metric cards, and responsive risk alerts.

---

## 🏛️ Pipeline Architecture

```
┌─────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│   Raw IBM Data  │ ────>│  Preprocessing Pipeline │ ────>│  Oversampling Imbalance  │
│  (7,043 rows)   │      │ (Clean, Encode, Scale)  │      │ (Stratified Resampling)  │
└─────────────────┘      └─────────────────────────┘      └──────────────────────────┘
                                                                        │
┌─────────────────┐      ┌─────────────────────────┐                    ▼
│  Streamlit App  │ <────│   SHAP Explainability   │ <──── ┌──────────────────────────┐
│   (5-Page UI)   │      │  (Global/Local SHAP)    │       │    Model Training &      │
└─────────────────┘      └─────────────────────────┘       │      Leaderboard         │
                                                           │ (LR, RF, XGBoost Tuning) │
                                                           └──────────────────────────┘
```

---

## 🗂️ Repository Structure

```directory
customer-churn-prediction/
├── data/
│   └── telco_churn.csv            # Telco Customer Churn dataset (IBM raw format)
├── models/                         # Saved artifacts (.pkl binaries & evaluation logs)
│   ├── scaler.pkl                 # Fitted StandardScaler object
│   ├── feature_names.pkl          # List of features output during preprocessing
│   ├── logistic_regression.pkl    # Trained Logistic Regression model
│   ├── random_forest.pkl          # Trained Random Forest classifier
│   ├── xgboost.pkl                # Trained XGBoost classifier
│   └── results.json               # Pre-compiled evaluation benchmarks
├── src/                            # Modular source code
│   ├── data_loader.py             # Loads dataset and runs validation rules
│   ├── preprocessing.py           # Handles clean, encoding, scaling, and oversampling
│   ├── train.py                   # Executes cross-validation, training, and model saving
│   ├── evaluate.py                # Houses Plotly charting functions for EDA/Metrics
│   └── explainability.py          # Wrapper functions for SHAP explainers and summaries
├── app/                            # Front-end deployment
│   └── streamlit_app.py           # Core multi-page app with styling and state tracking
├── notebooks/                      # Development & exploration
│   └── 01_full_walkthrough.ipynb  # Comprehensive Jupyter notebook guide
├── requirements.txt                # Python package list
└── README.md                       # Documentation index
```

---

## 💻 Setup & Local Installation

### 1. Clone & Initialize Workspace
First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Configure Virtual Environment
Create and activate a isolated environment:

**Windows (cmd / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
Ensure pip is updated and install project dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📥 Dataset Acquisition

The project is built on the popular **Telco Customer Churn** dataset (IBM). 

### Option A: Automate using Kaggle API
If you have configured the `kaggle` CLI tool:
```bash
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/
mv data/WA_Fn-UseC_-Telco-Customer-Churn.csv data/telco_churn.csv
rm telco-customer-churn.zip
```

### Option B: Direct/Manual Download
1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Create a folder named `data` in the project root.
3. Save the file inside the folder as `data/telco_churn.csv`.

---

## 🛠️ Data Pipeline Details

### Preprocessing Steps
Implemented inside [`src/preprocessing.py`](file:///C:/Users/hp/.gemini/antigravity-ide/scratch/customer-churn-prediction/src/preprocessing.py):
1.  **Numeric Conversions:** Coerces `TotalCharges` from string to float and imputes missing elements with the median.
2.  **Binary Mapping:** Maps `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, and `Churn` binary strings to `[0, 1]`.
3.  **One-Hot Encoding:** Encodes multi-category variables (e.g. `Contract`, `InternetService`, `PaymentMethod`, etc.) to prevent model assumptions about order.
4.  **Standard Scaling:** Scales continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to ensure zero mean and unit variance.
5.  **Oversampling:** Corrects severe class imbalance (~73% No / ~27% Yes) on the training set using stratified minority resampling, raising minority prevalence to 50/50.

---

## 🧠 Model Training

Execute the training pipeline to run cross-validation and serialize models:
```bash
python src/train.py
```

This script will:
*   Preprocess the raw data.
*   Fit Logistic Regression, Random Forest, and XGBoost models.
*   Compute 3-fold cross-validated ROC-AUC, precision, recall, and F1 scores.
*   Serialize trained models, scalers, and results parameters into the `models/` directory.

---

## 🖥️ Running the Streamlit Dashboard

Launch the interactive 5-page dashboard locally:
```bash
streamlit run app/streamlit_app.py
```
After executing, the dashboard will open automatically in your browser at `http://localhost:8501`. 

### Interactive Pages Walkthrough:
*   **🏠 Home:** Dashboard hub displaying total customer counts, global churn rates, average customer tenure, a preview of the dataset, and a performance leaderboard comparing all classifiers.
*   **📊 EDA (Exploratory Data Analysis):** Visualizes overall rates, contract category percentages, internet/payment distributions, histograms of charges, and a correlation matrix heatmap.
*   **🤖 Model Results:** Allows users to pick any trained model to view its test metrics, Plotly Heatmap confusion matrix, and feature importance bar chart.
*   **🔮 Predict (Live Scoring):** Provides an input form to create a custom customer profile, select a model, and compute an instant churn probability, risk category (Low/Medium/High), and actionable recommendations.
*   **🔍 Explain (SHAP Visualizations):** 
    *   *Global:* Displays a SHAP summary beeswarm plot demonstrating how individual feature levels shift base rates toward or away from churn.
    *   *Local:* Evaluates the exact profile submitted on the **Predict** page, generating a SHAP waterfall chart and returning the top five churn reasons formatted in plain English.

---

## 📊 Performance Benchmarks

The models demonstrate the following baseline evaluation on the test set:

| Model | ROC-AUC | F1 (Churn) | Recall (Churn) |
|---|---|---|---|
| **Logistic Regression** | ~0.84 | ~0.60 | ~0.75 |
| **Random Forest** | ~0.86 | ~0.62 | ~0.72 |
| **XGBoost** | **~0.88** | **~0.65** | **~0.78** |

*Tree-based classifiers (Random Forest & XGBoost) capture non-linear relationships more effectively, with XGBoost achieving the best balance of detection rate (Recall) and overall confidence (ROC-AUC).*

---

## 📄 License

This repository is open-sourced under the [MIT License](LICENSE).
