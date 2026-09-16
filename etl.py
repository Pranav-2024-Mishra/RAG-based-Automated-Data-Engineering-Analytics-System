"""
ETL Pipeline: Extract, Transform, Load helpers.
"""
import io
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def _parse_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Actual parsing, cached on the file's content + name. Streamlit reruns
    the whole script on every interaction (chat messages, button clicks
    anywhere in the app) — without this cache, a large file would get
    re-read from scratch on every single one of those reruns, which is
    what causes the slowdowns/hangs on heavy uploads."""
    name = filename.lower()
    buffer = io.BytesIO(file_bytes)
    if name.endswith(".csv"):
        df = pd.read_csv(buffer)
    elif name.endswith((".xlsx", ".xls")):
        try:
            # python-calamine is much faster than openpyxl on large Excel files
            buffer.seek(0)
            df = pd.read_excel(buffer, engine="calamine")
        except Exception:
            buffer.seek(0)
            df = pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    return df


def load_data(uploaded_file) -> pd.DataFrame:
    """Extract: Load data from an uploaded CSV or Excel file."""
    file_bytes = uploaded_file.getvalue()
    return _parse_file(file_bytes, uploaded_file.name)


def profile_data(df: pd.DataFrame) -> dict:
    """Generate a data profiling report: missing values, duplicates, outliers."""
    report = {}

    # --- Missing values ---
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2) if len(df) else missing
    missing_df = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
    })
    report["missing_values"] = missing_df[missing_df["missing_count"] > 0].reset_index(drop=True)

    # --- Duplicates ---
    report["duplicate_count"] = int(df.duplicated().sum())

    # --- Outliers (IQR method, numeric columns only) ---
    outlier_rows = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_out > 0:
            outlier_rows.append({
                "column": col, "outlier_count": n_out,
                "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
            })
    report["outliers"] = pd.DataFrame(outlier_rows)

    report["shape"] = df.shape
    report["dtypes"] = df.dtypes.astype(str).to_dict()
    return report


def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    missing_strategy: str = "drop",
    cap_outliers: bool = False,
) -> pd.DataFrame:
    """Transform: Clean the dataframe based on chosen strategies."""
    cleaned = df.copy()

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    if missing_strategy == "drop":
        cleaned = cleaned.dropna()
    elif missing_strategy == "fill_mean":
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        cleaned[num_cols] = cleaned[num_cols].fillna(cleaned[num_cols].mean())
    elif missing_strategy == "fill_median":
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        cleaned[num_cols] = cleaned[num_cols].fillna(cleaned[num_cols].median())
    elif missing_strategy == "fill_mode":
        for col in cleaned.columns:
            if cleaned[col].isnull().any():
                mode_val = cleaned[col].mode()
                if not mode_val.empty:
                    cleaned[col] = cleaned[col].fillna(mode_val.iloc[0])
    # "none" -> leave missing values as-is

    if cap_outliers:
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            q1, q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            cleaned[col] = cleaned[col].clip(lower, upper)

    return cleaned