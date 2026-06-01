# 📉 Customer Churn Prediction

A production-style ML project that predicts which telecom customers are likely to churn — and **explains why** using SHAP values. Built for a remote data science portfolio.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo
Deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud) or [Hugging Face Spaces](https://huggingface.co/spaces).

---

## 📌 What This Project Does

| Step | Description |
|---|---|
| EDA | Churn rate by contract, tenure, charges — interactive Plotly charts |
| Preprocessing | Encoding, scaling, SMOTE for class imbalance |
| Modelling | Logistic Regression, Random Forest, XGBoost — compared side by side |
| Evaluation | ROC-AUC, F1, Precision, Recall, Confusion Matrix |
| Explainability | SHAP global + local explanations per customer |
| Web App | 5-page Streamlit dashboard with live prediction |

---

## 🗂️ Project Structure

```
customer-churn-prediction/
├── data/
│   └── telco_churn.csv          # Download instructions below
├── models/                       # Saved after training
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── 01_full_walkthrough.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📥 Dataset

Download from Kaggle:
```bash
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/
mv data/WA_Fn-UseC_-Telco-Customer-Churn.csv data/telco_churn.csv
```

Or manually download from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 🧠 Train Models

```bash
python src/train.py
```

---

## 🖥️ Run the App

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`

---

## 📊 Results

| Model | ROC-AUC | F1 (Churn) | Recall (Churn) |
|---|---|---|---|
| Logistic Regression | ~0.84 | ~0.60 | ~0.75 |
| Random Forest | ~0.86 | ~0.62 | ~0.72 |
| XGBoost | ~0.88 | ~0.65 | ~0.78 |

---

## 🛠️ Tech Stack
pandas · numpy · scikit-learn · XGBoost · imbalanced-learn · SHAP · Plotly · Streamlit

---

## 📄 License
MIT
