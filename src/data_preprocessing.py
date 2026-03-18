import pandas as pd
import numpy as np
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Telco Churn dataset and perform initial type fixes."""
    df = pd.read_csv(filepath)
    # Fix TotalCharges (spaces stored as strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: handle missing values, duplicates, type conversions."""
    df = df.copy()

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Fill TotalCharges NaN with 0 (new customers, no charges yet)
    df['TotalCharges'].fillna(0, inplace=True)

    # Encode target column
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # SeniorCitizen is already 0/1
    # Strip whitespace from string cols
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    return df


def get_feature_columns(df: pd.DataFrame):
    """Return list of feature columns (drop ID and target)."""
    return [c for c in df.columns if c not in ['customerID', 'Churn']]


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all object columns (excluding customerID)."""
    df = df.copy()
    cat_cols = [c for c in df.select_dtypes(include='object').columns if c != 'customerID']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


def save_processed(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved processed data → {out_path}")
