"""
generate_sample_data.py
-----------------------
Creates a synthetic Telco-like CSV in data/raw/ so the Streamlit app
can demo BEFORE you download the real Kaggle dataset.

Usage:
    python generate_sample_data.py
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043

contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                              N, p=[0.55, 0.24, 0.21])
tenure = np.where(contracts == 'Month-to-month',
                  np.random.randint(1, 30, N),
                  np.where(contracts == 'One year',
                           np.random.randint(12, 48, N),
                           np.random.randint(24, 72, N)))

monthly_charges = np.where(
    np.random.choice(['DSL', 'Fiber optic', 'No'],
                     N, p=[0.34, 0.44, 0.22]) == 'Fiber optic',
    np.random.uniform(70, 110, N),
    np.random.uniform(20, 70, N)
).round(2)

total_charges = (monthly_charges * tenure + np.random.normal(0, 50, N)).clip(0).round(2)

churn_prob = (
    0.05
    + 0.35 * (contracts == 'Month-to-month')
    + 0.15 * (monthly_charges > 70)
    - 0.30 * (tenure > 36)
    + 0.05 * np.random.randn(N)
).clip(0, 1)
churn = np.where(np.random.rand(N) < churn_prob, 'Yes', 'No')

internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], N, p=[0.34, 0.44, 0.22])

df = pd.DataFrame({
    'customerID': [f'CUST-{i:05d}' for i in range(N)],
    'gender': np.random.choice(['Male', 'Female'], N),
    'SeniorCitizen': np.random.choice([0, 1], N, p=[0.84, 0.16]),
    'Partner': np.random.choice(['Yes', 'No'], N, p=[0.48, 0.52]),
    'Dependents': np.random.choice(['Yes', 'No'], N, p=[0.30, 0.70]),
    'tenure': tenure,
    'PhoneService': np.random.choice(['Yes', 'No'], N, p=[0.90, 0.10]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], N, p=[0.42, 0.48, 0.10]),
    'InternetService': internet_service,
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], N, p=[0.29, 0.50, 0.21]),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], N, p=[0.34, 0.44, 0.22]),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], N, p=[0.34, 0.44, 0.22]),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], N, p=[0.29, 0.49, 0.22]),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], N, p=[0.38, 0.40, 0.22]),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], N, p=[0.39, 0.39, 0.22]),
    'Contract': contracts,
    'PaperlessBilling': np.random.choice(['Yes', 'No'], N, p=[0.59, 0.41]),
    'PaymentMethod': np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        N, p=[0.34, 0.23, 0.22, 0.21]),
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn,
})

out = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
os.makedirs('data/raw', exist_ok=True)
df.to_csv(out, index=False)
print(f"✅  Synthetic dataset saved → {out}")
print(f"   Shape: {df.shape}  |  Churn rate: {(df['Churn']=='Yes').mean():.1%}")
