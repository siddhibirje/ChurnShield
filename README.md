# 🛡️ ChurnShield — Customer Churn Prediction Platform

> **Predict who will leave. Understand why. Act before it's too late.**

A full-stack machine learning platform built for the Indian telecom market — combining binary classification, SHAP explainability, RFM segmentation, survival analysis, and prescriptive ROI analytics into a single interactive web dashboard.

---

## 📊 Project Overview

ChurnShield answers three business-critical questions:

| Question | Solution |
|---|---|
| **Who** will churn? | 6 ML classifiers → best model auto-selected by AUC-ROC |
| **Why** will they churn? | SHAP explainability — per-customer feature attribution |
| **What** should we do? | Data-driven retention strategies with ₹ INR ROI estimates |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/cnread07/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Create environment (Anaconda recommended)
```bash
conda create -n churnshield python=3.11 -y
conda activate churnshield
pip install -r requirements.txt
pip install shap lifelines plotly
```

### 3. Get the dataset
Download the real Telco dataset from Kaggle:
```
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`

Or generate synthetic demo data instantly:
```bash
python generate_sample_data.py
```

### 4. Train all models
```bash
python train_models.py
```

### 5. Launch the dashboard
```bash
cd streamlit_app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🌐 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 **Overview** | Live KPIs, churn distribution, payment method analysis, tenure heatmaps |
| 🔍 **Customer Lookup** | Search any customer → churn probability gauge + SHAP explainability chart + retention recommendation |
| 👥 **Segments** | RFM lifecycle segments (Champions → Lost) + interactive K-Means clustering |
| 📈 **Model Performance** | ROC curves, PR curves, confusion matrix, feature importance, threshold optimizer |
| 🎯 **What-If Simulator** | Change attributes live → model re-predicts instantly + scenario comparison table |
| 💰 **ROI Calculator** | Data-derived lift rates, CLV distribution, ₹ campaign ROI with waterfall chart |

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
│
├── data/
│   ├── raw/                          ← Place Kaggle CSV here
│   └── processed/                    ← Auto-generated after training
│
├── models/                           ← Trained .pkl files
│   ├── best_model.pkl                ← Auto-selected best performer
│   ├── scaler.pkl                    ← StandardScaler
│   ├── feature_names.pkl             ← Feature column order
│   ├── model_comparison.csv          ← AUC / F1 results table
│   └── test_data.pkl                 ← Held-out test set
│
├── notebooks/                        ← 12 Jupyter notebooks (one per phase)
│
├── src/
│   ├── data_preprocessing.py         ← Cleaning pipeline
│   └── feature_engineering.py        ← RFM, engagement score, tenure groups
│
├── streamlit_app/
│   ├── app.py                        ← Landing page entry point
│   ├── pages/
│   │   ├── customer_lookup.py        ← SHAP + individual prediction
│   │   ├── model_performance.py      ← Evaluation + threshold optimizer
│   │   ├── overview.py               ← KPI dashboard
│   │   ├── roi_calculator.py         ← Retention ROI engine
│   │   ├── segments.py               ← RFM + K-Means
│   │   └── whatif_simulator.py       ← Live counterfactual simulator
│   └── utils/
│       ├── data_loader.py            ← Cached data/model loading
│       └── styles.py                 ← Dark/light theme CSS
│
├── generate_sample_data.py           ← Synthetic data generator
├── train_models.py                   ← Full training pipeline
└── requirements.txt
```

---

## 🧠 ML Pipeline

### Dataset
- **Source:** IBM Telco Customer Churn — Kaggle
- **Size:** 7,043 customers · 21 raw features · ~26% churn rate
- **Target:** `Churn` (binary — Yes/No)

### Data Cleaning (`src/data_preprocessing.py`)
1. Load CSV with `pd.read_csv`
2. Fix `TotalCharges` — stored as string, converted to numeric
3. Drop duplicates
4. Impute NaN TotalCharges with 0 (new customers)
5. Encode target: `Yes → 1`, `No → 0`
6. Strip whitespace from all string columns

### Feature Engineering (`src/feature_engineering.py`)

| Feature | Method | Purpose |
|---|---|---|
| `tenure_group` | `pd.cut()` into 4 lifecycle buckets | Captures early vs mature customers |
| `engagement_score` | Count of "Yes" across 8 service columns | Proxy for stickiness |
| `charges_per_month` | TotalCharges ÷ tenure | Normalised spend |
| `recency / frequency / monetary` | RFM proxy from dataset columns | Customer value framework |
| `RFM_Score` | `pd.qcut()` quintile scoring (1–5 per R, F, M) | Data-driven, adapts to distribution |
| `RFM_Segment` | Score → Champions / Loyal / At-Risk / Lost | Actionable segmentation labels |

### Models Trained (`train_models.py`)

| Model | Key Parameters | Target AUC-ROC |
|---|---|---|
| Logistic Regression | `C=1.0, class_weight='balanced'` | 0.78+ |
| Decision Tree | `max_depth=6, min_samples_split=30` | 0.75+ |
| Random Forest | `n_estimators=200, max_depth=10` | 0.83+ |
| XGBoost | `n_estimators=300, lr=0.05, scale_pos_weight` | 0.85+ |
| LightGBM | `n_estimators=300, lr=0.05` | 0.85+ |
| Voting Ensemble | Soft vote: RF + XGBoost + LightGBM | 0.86+ |

**Training setup:**
- 80/20 stratified train/test split
- `StandardScaler` applied to all features
- `class_weight='balanced'` to handle 26% churn imbalance
- Best model auto-saved as `best_model.pkl` by AUC-ROC

---

## 🔍 SHAP Explainability

Each customer lookup includes a SHAP waterfall chart explaining **why** the model assigned that specific churn probability:

- 🔴 **Red bars** — features pushing churn probability **up**
- 🔵 **Blue bars** — features pushing churn probability **down**
- Plain-English summary of top risk factors below the chart

SHAP uses `TreeExplainer` for tree-based models and `LinearExplainer` for Logistic Regression. Values represent each feature's marginal contribution to the prediction relative to the model's baseline.

---

## 💰 ROI Calculator — What Makes It Data-Driven

Retention lift rates are **not hardcoded** — they are derived from actual churn rate differences between contract types in the Telco dataset:

```
Lift (Premium)  = churn_rate(Month-to-month) − churn_rate(Two year)
Lift (Standard) = churn_rate(Month-to-month) − churn_rate(One year)
```

CLV formula:
```
CLV = MonthlyCharges × avg_lifetime_months × gross_margin × INR_rate
avg_lifetime_months = 1 / actual_churn_rate  (derived from dataset)
```

Campaign costs are the only business input — exposed as adjustable sliders.

---

## 🎯 What-If Simulator — Scenario Comparison

The simulator runs **actual model inference** on modified customer profiles — not rules or heuristics. The scenario comparison table shows the real `predict_proba()` output for:

- Upgrading to a two-year contract
- Adding tech support
- Switching to the lowest-churn payment method (picked from data)
- Retaining the customer 12 more months
- Adding all protection services

---

## 📈 Novel Contributions

- **SHAP per-customer explainability** — why this customer was flagged
- **Data-derived lift rates** — ROI built from actual dataset churn differentials
- **Live scenario comparison** — real model inference on counterfactuals
- **Threshold optimizer** — precision/recall/F1 tradeoff chart across all thresholds
- **Full pipeline** — EDA → Cleaning → Feature Engineering → 6 Models → Dashboard → Action

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web framework | Streamlit |
| ML models | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Survival analysis | lifelines |
| Visualisation | Plotly |
| Data layer | Pandas, NumPy |
| Model storage | pickle (.pkl) |

---

## 👥 Team

| Member | Responsibility |
|---|---|
| Person 1 | Data Cleaning · Survival Analysis |
| Person 2 | EDA · Baseline Models |
| Person 3 | Feature Engineering · Ensemble Models |
| Person 4 | Streamlit Dashboard · ROI Calculator |

---

## 📚 Dataset

**Telco Customer Churn** — IBM Watson / Kaggle  
7,043 customers · 21 features · ~26% churn rate  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 📌 Notes

- Models are trained once and served via `.pkl` files — the app never retrains at runtime
- All churn probabilities come from `model.predict_proba()` on the real trained model
- RFM segmentation uses `pd.qcut()` — thresholds adapt to your data distribution, not hardcoded
- The synthetic data generator (`generate_sample_data.py`) exists for demo purposes only — use the real Kaggle CSV for accurate results

---

*Built for ML Coursework · Telecom Churn Prediction · Indian Market Focus*
