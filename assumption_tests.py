# assumption_tests.py

import pandas as pd

try:
    import statsmodels.stats.api as sms
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False


def regression_assumptions(model, df, X):

    if not STATS_MODELS_AVAILABLE:
        return {"Error": "statsmodels not installed"}

    results = {}

    try:
        test = sms.het_breuschpagan(model.resid, model.model.exog)
        results["Heteroskedasticity p-value"] = test[1]
    except:
        results["Heteroskedasticity"] = "Test failed"

    try:
        X_data = df[X].dropna()

        vif = pd.DataFrame()
        vif["Variable"] = X_data.columns
        vif["VIF"] = [
            variance_inflation_factor(X_data.values, i)
            for i in range(len(X_data.columns))
        ]

        results["VIF"] = vif
    except:
        results["VIF"] = "Could not compute"

    return results
