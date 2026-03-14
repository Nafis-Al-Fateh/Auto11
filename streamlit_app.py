import streamlit as st
import pandas as pd

from profiler import profile_data
from recommendation import recommend_methods
from analysis import run_correlation, run_regression
from visualization import plot_corr_heatmap, regression_plots
from assumption_tests import regression_assumptions
from report_generator import generate_report
from code_generator import generate_regression_code
from ai_interpreter import interpret_regression

st.set_page_config(page_title="AI Statistical Assistant", layout="wide")

st.title("AI Statistical Research Assistant")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv","xlsx"])

if uploaded_file:

    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset Loaded")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Dataset Profiling
    # -----------------------------

    st.header("Dataset Diagnostics")

    profile = profile_data(df)

    st.write(profile)

    # -----------------------------
    # Method Recommendation
    # -----------------------------

    st.header("Recommended Methods")

    methods = recommend_methods(df)

    for m in methods:
        st.write("✔", m)

    # -----------------------------
    # Variable Selection
    # -----------------------------

    numeric_cols = df.select_dtypes(include='number').columns

    st.header("Select Variables")

    y = st.selectbox("Dependent Variable", numeric_cols)
    X = st.multiselect("Independent Variables", numeric_cols)

    # -----------------------------
    # Correlation
    # -----------------------------

    if st.button("Run Correlation"):

        corr = run_correlation(df)

        st.write(corr)

        fig = plot_corr_heatmap(corr)

        st.plotly_chart(fig)

    # -----------------------------
    # Regression
    # -----------------------------

    if st.button("Run Regression"):

        results = run_regression(df, y, X)

        st.text(results.summary())

        # Visualization
        fig1, fig2 = regression_plots(results)

        st.pyplot(fig1)
        st.pyplot(fig2)

        # Assumptions
        st.header("Regression Assumptions")

        tests = regression_assumptions(results, df, X)

        st.write(tests)

        # AI Interpretation
        st.header("AI Interpretation")

        explanation = interpret_regression(results)

        st.write(explanation)

        # Reproducible Code
        st.header("Generated Python Code")

        code = generate_regression_code(y, X)

        st.code(code)

        # Report
        if st.button("Generate Research Report"):

            path = generate_report(results, explanation)

            with open(path, "rb") as file:
                st.download_button(
                    "Download Report",
                    file,
                    file_name="analysis_report.docx"
                )
