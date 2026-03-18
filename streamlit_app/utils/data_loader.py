"""Shared data loading & model cache — used by all Streamlit pages."""
import os, sys, pickle
import pandas as pd
import numpy as np
import streamlit as st

# Allow imports from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_PATH  = os.path.join(ROOT, 'data', 'processed', 'cleaned_features.csv')
MODEL_DIR  = os.path.join(ROOT, 'models')
RAW_PATH   = os.path.join(ROOT, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')


@st.cache_data(show_spinner=False)
def load_main_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    # Fallback: load raw and do minimal processing
    from src.data_preprocessing import load_data, clean_data
    from src.feature_engineering import apply_all_features
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = apply_all_features(df)
    return df


@st.cache_resource(show_spinner=False)
def load_model(name: str = 'best_model'):
    path = os.path.join(MODEL_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_scaler():
    path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_feature_names():
    path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_test_data():
    path = os.path.join(MODEL_DIR, 'test_data.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    path = os.path.join(MODEL_DIR, 'model_comparison.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)


def models_trained() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, 'best_model.pkl'))


def data_available() -> bool:
    return os.path.exists(DATA_PATH) or os.path.exists(RAW_PATH)
