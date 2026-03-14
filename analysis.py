# analysis.py

import pandas as pd
import streamlit as st

# Safe import of statsmodels
try:
    import statsmodels.api as sm
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False


@st.cache_data
def run_correlation(df: pd.DataFrame):

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least two numeric variables for correlation.")

    corr = numeric_df.corr()

    return corr


@st.cache_data
def run_regression(df: pd.DataFrame, y: str, X: list):

    if not STATS_MODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is not installed. Please add statsmodels to requirements.txt"
        )

    if len(X) == 0:
        raise ValueError("Select at least one independent variable.")

    data = df[[y] + X].dropna()

    y_data = data[y]
    X_data = data[X]

    X_data = sm.add_constant(X_data)

    model = sm.OLS(y_data, X_data).fit()

    return model
