import numpy as np
import pandas as pd


def calculate_weighted_score(pdf: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Dynamically calculate weighted score 'wgt' from given columns and weights.

    Parameters:
        pdf: pandas DataFrame with edge data
        weights: dict mapping column names to weight values
                 e.g. {"delta_time": -0.2}

    Returns:
        pdf with 'wgt' column added
    """
    pdf['wgt'] = 0

    for col, weight in weights.items():
        if col in pdf.columns:
            pdf['wgt'] += pdf[col] * weight
        else:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")

    return pdf


def scale_wgt(pdf: pd.DataFrame, scaler: str = "scale_and_round", scale_factor: int = 1000) -> pd.DataFrame:
    """
    Scale the 'wgt' column for cuGraph compatibility (must be int32).

    Parameters:
        pdf: pandas DataFrame with 'wgt' column
        scaler: one of ["scale_and_round", "retain_float", "normalize_range", "log_transform"]
        scale_factor: multiplier for scaling

    Returns:
        pdf with scaled 'wgt' column
    """
    scaling_plans = ["scale_and_round", "retain_float", "normalize_range", "log_transform"]

    if scaler == scaling_plans[0]:
        pdf['wgt'] = (pdf['wgt'] * scale_factor).astype('int32')

    elif scaler == scaling_plans[1]:
        pdf['wgt'] = pdf['wgt'].round(2)

    elif scaler == scaling_plans[2]:
        min_wgt = pdf['wgt'].min()
        max_wgt = pdf['wgt'].max()
        pdf['wgt'] = (((pdf['wgt'] - min_wgt) / (max_wgt - min_wgt)) * scale_factor).astype('int32')

    else:  # log_transform
        pdf['wgt'] = np.log1p(pdf['wgt'])
        pdf['wgt'] = (pdf['wgt'] * scale_factor).astype('int32')

    return pdf


def compute_delta_time(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta_time between same (src, dst) pairs.

    Parameters:
        pdf: pandas DataFrame with 'src', 'dst', 'timestamp' columns

    Returns:
        pdf with 'delta_time' column (in seconds)
    """
    pdf['delta_time'] = pdf.groupby(['src', 'dst'])['timestamp'].diff()
    pdf['delta_time'] = pdf['delta_time'].dt.seconds + pdf['delta_time'].dt.nanoseconds / 1e9
    pdf['delta_time'] = pdf['delta_time'].fillna(0)
    return pdf


def normalize_wgt(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalize 'wgt' and shift to positive values.

    Parameters:
        pdf: pandas DataFrame with 'wgt' column

    Returns:
        pdf with normalized 'wgt' column
    """
    std = pdf['wgt'].std()

    if std == 0 or pd.isna(std):
        pdf['wgt'] = 1.0
        return pdf

    pdf['wgt_zscore'] = (pdf['wgt'] - pdf['wgt'].mean()) / pdf['wgt'].std()
    min_zscore = pdf['wgt_zscore'].min()
    pdf['wgt'] = pdf['wgt_zscore'] + abs(min_zscore) + 1
    return pdf
