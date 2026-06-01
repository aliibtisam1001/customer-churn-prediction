# Data Directory

This folder is intentionally empty in the repo.

## Download the dataset

**Option 1 — Kaggle CLI:**
```bash
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/
mv data/WA_Fn-UseC_-Telco-Customer-Churn.csv data/telco_churn.csv
```

**Option 2 — Manual:**  
Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Save as `data/telco_churn.csv`

## Expected file after download
```
data/
└── telco_churn.csv     # 7,043 rows × 21 columns
```
