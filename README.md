# 🛡️ ChurnShield — Customer Churn Prediction Platform

> **ML Coursework Project** | Telecom Customer Churn Analysis for the Indian Market  
> Integrates Binary Classification · Survival Analysis · Customer Segmentation · Prescriptive Analytics

---

## 📊 Project Overview

ChurnShield predicts **which** customers will churn, **when** they'll churn, and **what to do** about it — combining machine learning, RFM segmentation, and survival analysis into a single interactive web dashboard.

| Phase | Description | Status |
|-------|-------------|--------|
| EDA & Cleaning | Exploratory Data Analysis | ✅ |
| Feature Engineering | RFM, Tenure Groups, Engagement Score | ✅ |
| Customer Segmentation | K-Means + RFM Segments | ✅ |
| Classification Models | LR, DT, RF, XGBoost, LightGBM, Ensemble | ✅ |
| Survival Analysis | Kaplan-Meier + Cox PH | 🔄 Week 4 |
| CLV Analysis | Customer Lifetime Value | 🔄 Week 4 |
| Streamlit Dashboard | Interactive Web App | ✅ |

---

## 🚀 Quick Start

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the Dataset
**Option A — Kaggle (recommended, real data):**
```bash
# Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Download: WA_Fn-UseC_-Telco-Customer-Churn.csv
# Place in: data/raw/
```

**Option B — Generate synthetic data (demo instantly):**
```bash
python generate_sample_data.py
```

### 3. Train All Models
```bash
python train_models.py
```

### 4. Launch the Website
```bash
cd streamlit_app
streamlit run app.py
```

---

## 🌐 Website Pages

| Page | Description |
|------|-------------|
| 🏠 **Overview** | Churn KPIs, distribution charts, heatmaps |
| 🔍 **Customer Lookup** | Search any customer → get churn risk + recommendation |
| 👥 **Segments** | RFM segments + interactive K-Means clustering |
| 📈 **Model Performance** | ROC, PR curves, confusion matrix, threshold optimizer |
| 🎯 **What-If Simulator** | Change attributes, watch churn probability update live |
| 💰 **ROI Calculator** | Simulate retention campaigns with real ₹ ROI numbers |

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
├── data/
│   ├── raw/              ← Place Kaggle CSV here
│   └── processed/        ← Auto-generated after training
├── notebooks/            ← 12 Jupyter notebooks (one per phase)
├── models/               ← Trained .pkl model files
├── src/                  ← Reusable Python modules
├── streamlit_app/        ← The website
│   ├── app.py            ← Entry point
│   ├── pages/            ← One file per page
│   └── utils/            ← Shared helpers
├── generate_sample_data.py
├── train_models.py
└── requirements.txt
```

---

## 📈 Models & Performance Targets

| Model | Target AUC-ROC |
|-------|---------------|
| Logistic Regression | 0.78+ |
| Decision Tree | 0.75+ |
| Random Forest | 0.83+ |
| XGBoost | 0.85+ |
| LightGBM | 0.85+ |
| Voting Ensemble | **0.86+** |

---

## 🧠 Novel Contributions
1. **Survival Analysis** — Predicts WHEN customers churn, not just IF
2. **SHAP Explainability** — Why did this customer get flagged?  
3. **What-If Simulator** — Interactive counterfactuals for each customer
4. **₹ ROI Calculator** — Prescriptive analytics with Indian rupee economics
5. **Full Pipeline** — EDA → Segmentation → Classification → Action in one platform

---

## 👥 Team
> Fill in team member names here

| Member | Responsibility |
|--------|---------------|
| Person 1 | Data Cleaning, Survival Analysis |
| Person 2 | EDA, Baseline Models |
| Person 3 | Feature Engineering, Ensemble Models |
| Person 4 | Streamlit Dashboard, ROI Calculator |

---

## 📚 Dataset
[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
7,043 customers · 21 features · ~26% churn rate
