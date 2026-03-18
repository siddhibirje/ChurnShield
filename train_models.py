"""
train_models.py
----------------
Run this script ONCE to train all models and save them as .pkl files.

Usage:
    python train_models.py

Note: XGBoost on macOS requires libomp: brew install libomp
      If not installed, XGBoost is automatically skipped.
"""
import os
import pickle
import shutil
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score
)

# Optional imports — gracefully skip if not available
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    print("⚠️  XGBoost not available (install libomp: brew install libomp). Skipping.")

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False
    print("⚠️  LightGBM not available. Skipping.")

import sys
sys.path.append(os.path.dirname(__file__))
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import apply_all_features

warnings.filterwarnings('ignore')

DATA_PATH = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('data/processed', exist_ok=True)


# ─── 1. Load & Prepare Data ────────────────────────────────────────────────
print("Loading data...")
df_raw = load_data(DATA_PATH)
df_clean = clean_data(df_raw)
df_feat = apply_all_features(df_clean)

df_feat.to_csv('data/processed/cleaned_features.csv', index=False)
print(f"  Rows: {len(df_feat):,}  |  Churn rate: {df_feat['Churn'].mean():.1%}")

# ─── 2. Feature Prep ───────────────────────────────────────────────────────
DROP_COLS = ['customerID', 'Churn', 'tenure_group', 'RFM_Segment',
             'recency', 'frequency', 'monetary', 'R_Score', 'F_Score', 'M_Score']
X_raw = df_feat[[c for c in df_feat.columns if c not in DROP_COLS]]
y = df_feat['Churn']

X = pd.get_dummies(X_raw, drop_first=True)

with open(f'{MODEL_DIR}/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open(f'{MODEL_DIR}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"  Features: {X.shape[1]}")

# ─── 3. Train / Test Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

results = {}
trained_models = {}


def train_and_evaluate(name, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc_val = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)
    results[name] = {
        'auc_roc': round(auc_val, 4),
        'avg_precision': round(ap, 4),
        'accuracy': round(report['accuracy'], 4),
        'precision_churn': round(report['1']['precision'], 4),
        'recall_churn': round(report['1']['recall'], 4),
        'f1_churn': round(report['1']['f1-score'], 4),
    }
    trained_models[name] = model
    print(f"  [{name:25s}] AUC={auc_val:.4f} | F1={report['1']['f1-score']:.4f}")
    return model


# ─── 4. Train Models ───────────────────────────────────────────────────────
print("\nTraining models...")

lr = train_and_evaluate('Logistic Regression',
    LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced'))
with open(f'{MODEL_DIR}/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)

dt = train_and_evaluate('Decision Tree',
    DecisionTreeClassifier(max_depth=6, min_samples_split=30, random_state=42, class_weight='balanced'))
with open(f'{MODEL_DIR}/decision_tree.pkl', 'wb') as f:
    pickle.dump(dt, f)

rf = train_and_evaluate('Random Forest',
    RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                           random_state=42, class_weight='balanced', n_jobs=-1))
with open(f'{MODEL_DIR}/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)

xgb = None
if _HAS_XGB:
    scale_pos = int((y == 0).sum() / (y == 1).sum())
    xgb = train_and_evaluate('XGBoost',
        XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                      scale_pos_weight=scale_pos, random_state=42,
                      eval_metric='logloss', n_jobs=-1))
    with open(f'{MODEL_DIR}/xgboost.pkl', 'wb') as f:
        pickle.dump(xgb, f)

lgbm = None
if _HAS_LGBM:
    lgbm = train_and_evaluate('LightGBM',
        LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                       class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))
    with open(f'{MODEL_DIR}/lightgbm.pkl', 'wb') as f:
        pickle.dump(lgbm, f)

# Voting Ensemble (use whichever models are available)
ensemble_parts = [('rf', rf)]
if lgbm is not None:
    ensemble_parts.append(('lgbm', lgbm))
if xgb is not None:
    ensemble_parts.append(('xgb', xgb))

if len(ensemble_parts) >= 2:
    voting = train_and_evaluate('Voting Ensemble',
        VotingClassifier(ensemble_parts, voting='soft', n_jobs=-1))
    with open(f'{MODEL_DIR}/voting_ensemble.pkl', 'wb') as f:
        pickle.dump(voting, f)

# ─── 5. Save Results & Best Model ─────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['auc_roc'])
model_file_map = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl',
    'LightGBM': 'lightgbm.pkl',
    'Voting Ensemble': 'voting_ensemble.pkl',
}
shutil.copy(f"{MODEL_DIR}/{model_file_map[best_name]}", f"{MODEL_DIR}/best_model.pkl")

results_df = pd.DataFrame(results).T
results_df.to_csv(f'{MODEL_DIR}/model_comparison.csv')

with open(f'{MODEL_DIR}/test_data.pkl', 'wb') as f:
    pickle.dump({'X_test': X_test, 'y_test': y_test.values,
                 'X_train': X_train, 'y_train': y_train.values}, f)

print(f"\n✅  Best model: {best_name} (AUC={results[best_name]['auc_roc']})")
print(f"   Saved to models/best_model.pkl")
print("\nModel Comparison:")
print(results_df.to_string())
