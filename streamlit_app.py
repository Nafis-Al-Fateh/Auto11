# streamlit_app.py

import streamlit as st
import pandas as pd

from profiler import profile_data
from recommendation import recommend_methods
from analysis import run_correlation, run_regression
from visualization import plot_corr_heatmap, regression_plots
from assumption_tests import regression_assumptions
from code_generator import generate_regression_code
from report_generator import generate_report

# Optional AI module
try:
    from ai_interpreter import interpret_regression
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Statistical Research Assistant",
    layout="wide"
)

st.title("AI Statistical Research Assistant")

st.write(
"""
Upload a dataset to automatically explore, analyze, and interpret results.
"""
)

# --------------------------------------------------
# Dataset upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.success("Dataset loaded successfully")

    except Exception as e:

        st.error(f"Failed to load dataset: {e}")
        st.stop()

    # --------------------------------------------------
    # Dataset preview
    # --------------------------------------------------

    st.header("Dataset Preview")

    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    # --------------------------------------------------
    # Dataset profiling
    # --------------------------------------------------

    st.header("Dataset Diagnostics")

    try:

        profile = profile_data(df)

        st.json(profile)

    except Exception as e:

        st.warning(f"Profiling failed: {e}")

    # --------------------------------------------------
    # Method recommendation
    # --------------------------------------------------

    st.header("Recommended Methods")

    try:

        methods = recommend_methods(df)

        for m in methods:
            st.write("✔", m)

    except Exception as e:

        st.warning(f"Recommendation failed: {e}")

    # --------------------------------------------------
    # Variable selection
    # --------------------------------------------------

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:

        st.warning("Dataset must contain at least two numeric variables.")
        st.stop()

    st.header("Variable Selection")

    y = st.selectbox(
        "Dependent Variable",
        numeric_cols
    )

    X = st.multiselect(
        "Independent Variables",
        [col for col in numeric_cols if col != y]
    )

    # --------------------------------------------------
    # Correlation Analysis
    # --------------------------------------------------

    st.header("Correlation Analysis")

    if st.button("Run Correlation"):

        try:

            corr = run_correlation(df)

            st.dataframe(corr)

            fig = plot_corr_heatmap(corr)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:

            st.error(f"Correlation failed: {e}")

    # --------------------------------------------------
    # Regression Analysis
    # --------------------------------------------------

    st.header("Regression Analysis")

    if st.button("Run Regression"):

        try:

            results = run_regression(df, y, X)

            st.subheader("Model Summary")

            st.text(results.summary())

            # --------------------------------------------------
            # Visualization
            # --------------------------------------------------

            st.subheader("Regression Diagnostics")

            fig1, fig2 = regression_plots(results)

            st.pyplot(fig1)
            st.pyplot(fig2)

            # --------------------------------------------------
            # Assumption tests
            # --------------------------------------------------

            st.subheader("Assumption Checks")

            tests = regression_assumptions(results, df, X)

            st.write(tests)

            # --------------------------------------------------
            # AI interpretation
            # --------------------------------------------------

            if AI_AVAILABLE:

                st.subheader("AI Interpretation")

                try:

                    explanation = interpret_regression(results)

                    st.write(explanation)

                except Exception as e:

                    st.warning(f"AI interpretation failed: {e}")
                    explanation = "AI interpretation unavailable."

            else:

                explanation = "AI module not installed."

            # --------------------------------------------------
            # Code generation
            # --------------------------------------------------

            st.subheader("Reproducible Python Code")

            code = generate_regression_code(y, X)

            st.code(code, language="python")

            # --------------------------------------------------
            # Report generation
            # --------------------------------------------------

            st.subheader("Generate Research Report")

            if st.button("Create Report"):

                try:

                    path = generate_report(results, explanation)

                    with open(path, "rb") as file:

                        st.download_button(
                            "Download Report",
                            file,
                            file_name="analysis_report.docx"
                        )

                except Exception as e:

                    st.error(f"Report generation failed: {e}")

        except Exception as e:

            st.error(f"Regression analysis failed: {e}")

else:

    st.info("Upload a dataset to begin analysis.")
