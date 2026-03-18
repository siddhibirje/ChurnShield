import pandas as pd
import numpy as np


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket tenure into meaningful groups."""
    df = df.copy()
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12 months', '12-24 months', '24-48 months', '48+ months'],
        right=True
    )
    return df


def create_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of 'Yes' services per customer (engagement proxy)."""
    df = df.copy()
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    existing = [c for c in service_cols if c in df.columns]
    df['engagement_score'] = df[existing].apply(
        lambda row: (row == 'Yes').sum(), axis=1
    )
    return df


def create_charges_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate inferred average monthly charge (guards against tenure=0)."""
    df = df.copy()
    df['charges_per_month'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )
    return df


def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy RFM from available Telco features:
      Recency   → inverse of tenure (long tenure = more recent interaction history)
      Frequency → engagement_score (# services)
      Monetary  → TotalCharges
    """
    df = df.copy()
    max_tenure = df['tenure'].max()

    # Recency: lower tenure → higher recency score
    df['recency'] = max_tenure - df['tenure']

    # Frequency: must run after create_engagement_score
    if 'engagement_score' not in df.columns:
        df = create_engagement_score(df)
    df['frequency'] = df['engagement_score']

    # Monetary
    df['monetary'] = df['TotalCharges']

    return df


def score_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Assign 1-5 quintile scores for R, F, M."""
    df = df.copy()
    # Recency: lower recency value = better → reverse labels
    df['R_Score'] = pd.qcut(df['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    df['F_Score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    df['M_Score'] = pd.qcut(df['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

    df['R_Score'] = df['R_Score'].astype(int)
    df['F_Score'] = df['F_Score'].astype(int)
    df['M_Score'] = df['M_Score'].astype(int)
    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']

    # Segment labels
    def label_segment(score):
        if score >= 12:
            return 'Champions'
        elif score >= 9:
            return 'Loyal Customers'
        elif score >= 6:
            return 'Potential Loyalists'
        elif score >= 4:
            return 'At Risk'
        else:
            return 'Lost'

    df['RFM_Segment'] = df['RFM_Score'].apply(label_segment)
    return df


def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = create_tenure_groups(df)
    df = create_engagement_score(df)
    df = create_charges_per_month(df)
    df = create_rfm_features(df)
    df = score_rfm(df)
    return df
