"""
AGRI-SMART DZ - Price prediction (Linear Regression, Random Forest, naive baseline).

Run:  python src/price_prediction.py      -> trains, evaluates, saves models/price_model_h{7,30}.pkl
Metrics are computed on a chronological hold-out set (2026). Nothing is claimed beyond that.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from . import data_processing as dp
else:
    import data_processing as dp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_VERSION = "rf_v2"


def load_model_bundle(horizon):
    model_path = os.path.join(MODEL_DIR, f"price_model_h{horizon}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for the {horizon}-day horizon has not been trained.")

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            bundle = joblib.load(model_path)
        except InconsistentVersionWarning as exc:
            raise RuntimeError(
                f"Model for the {horizon}-day horizon was trained with a different scikit-learn version."
            ) from exc

    if bundle.get("version") != MODEL_VERSION or bundle.get("sklearn_version") != sklearn.__version__:
        raise RuntimeError(
            f"Model for the {horizon}-day horizon is outdated; retrain it with the current environment."
        )
    return bundle


def model_is_current(horizon):
    try:
        load_model_bundle(horizon)
    except (FileNotFoundError, RuntimeError):
        return False
    return True


def _metrics(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def train_and_evaluate(horizon, df_clean):
    df, X, y = dp.build_supervised(df_clean, horizon)
    Xtr, Xte, ytr, yte, dtr, dte = dp.chrono_split(df, X, y, horizon=horizon)

    # The model predicts a RATIO; we convert back to DZD/kg for metrics.
    p0 = dte["price"].to_numpy()
    y_true = yte.to_numpy() * p0
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                               n_jobs=-1, random_state=42).fit(Xtr, ytr)

    results = {
        "Naive (price today)": _metrics(y_true, p0),
        "Linear Regression": _metrics(y_true, lin.predict(Xte) * p0),
        "Random Forest": _metrics(y_true, rf.predict(Xte) * p0),
    }
    # Uncertainty: 90% interval of the ratio residuals of the Random Forest (test set)
    resid = yte.to_numpy() - rf.predict(Xte)
    bundle = {
        "model": rf, "columns": list(X.columns), "horizon": horizon, "version": MODEL_VERSION,
        "sklearn_version": sklearn.__version__,
        "resid_q05": float(np.quantile(resid, 0.05)), "resid_q95": float(np.quantile(resid, 0.95)),
        "metrics": results,
        "importances": dict(sorted(zip(X.columns, rf.feature_importances_), key=lambda t: -t[1])[:8]),
        "trained_on": "SYNTHETIC_DEMO",
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, os.path.join(MODEL_DIR, f"price_model_h{horizon}.pkl"))
    return results, bundle


def predict_latest(horizon, df_clean, region=None, stock_adjustment_tonnes=0.0):
    """Predict the price `horizon` days after the latest date, per region, with a 90% interval."""
    bundle = load_model_bundle(horizon)
    forecast_input = df_clean.copy()
    if stock_adjustment_tonnes:
        if region is None:
            raise ValueError("A region is required when applying a stock adjustment.")
        region_rows = forecast_input[forecast_input["region"] == region]
        if region_rows.empty:
            raise ValueError(f"No data is available for region: {region}")
        latest_index = region_rows["date"].idxmax()
        adjusted_stock = float(forecast_input.loc[latest_index, "stock"]) + float(stock_adjustment_tonnes)
        if adjusted_stock < 0:
            raise ValueError("The stock adjustment cannot reduce available stock below zero.")
        forecast_input.loc[latest_index, "stock"] = adjusted_stock

    feats = dp.add_features(forecast_input)
    last = feats.sort_values("date").groupby("region").tail(1).dropna(subset=dp.FEATURES_NUM)
    if region:
        last = last[last["region"] == region]
    X = pd.concat([last[dp.FEATURES_NUM], pd.get_dummies(last["region"], prefix="reg", dtype=float)], axis=1)
    X = X.reindex(columns=bundle["columns"], fill_value=0.0)
    ratio = bundle["model"].predict(X)
    p0 = last["price"].to_numpy()
    pred = ratio * p0
    out = last[["region", "date", "price"]].copy()
    out["predicted_price"] = pred.round(2)
    out["low_90"] = ((ratio + bundle["resid_q05"]) * p0).round(2)
    out["high_90"] = ((ratio + bundle["resid_q95"]) * p0).round(2)
    out["target_date"] = out["date"] + pd.Timedelta(days=horizon)
    out["trend"] = np.where(out["predicted_price"] > out["price"] * 1.03, "UP",
                            np.where(out["predicted_price"] < out["price"] * 0.97, "DOWN", "STABLE"))
    return out.reset_index(drop=True), bundle


if __name__ == "__main__":
    clean = dp.load_clean()
    for h in dp.HORIZONS:
        res, b = train_and_evaluate(h, clean)
        print(f"\n=== Horizon {h} days (chronological test = 2026) ===")
        print(pd.DataFrame(res).T)
        print("Top features:", {k: round(v, 3) for k, v in b["importances"].items()})
    print("\nNOTE: trained on SYNTHETIC_DEMO data - these scores say nothing about real Algerian markets.")
    p, _ = predict_latest(7, clean)
    print(p)
