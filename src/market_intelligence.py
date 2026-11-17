"""Leakage-aware market forecasting and explainable, evidence-based risk signals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_CLASSES = {
    "Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=160, min_samples_leaf=3, max_depth=10, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.04, loss="huber", random_state=42
    ),
}
FEATURES = (
    "price_lag_1",
    "price_lag_7",
    "price_lag_30",
    "price_mean_7",
    "price_mean_30",
    "stock_ratio_30",
    "price_change_7",
    "stock_change_7",
    "month_sin",
    "month_cos",
)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _daily_panel(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "region", "price"}
    if observations.empty or not required.issubset(observations.columns):
        return pd.DataFrame()
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if "unit" not in frame:
        frame["unit"] = ""
    frame["unit"] = frame["unit"].fillna("").astype(str)
    if "quantity" in frame:
        frame["stock"] = pd.to_numeric(frame["quantity"], errors="coerce")
    elif "available_quantity" in frame:
        frame["stock"] = pd.to_numeric(frame["available_quantity"], errors="coerce")
    else:
        frame["stock"] = np.nan
    frame = frame.dropna(subset=["date", "region", "price"])
    if frame.empty:
        return frame
    frame = (
        frame.groupby(["region", "unit", "date"], as_index=False)
        .agg(price=("price", "median"), stock=("stock", "median"))
        .sort_values(["region", "date"])
    )
    daily = []
    for (region, unit), history in frame.groupby(["region", "unit"]):
        history = history.set_index("date").asfreq("D")
        history["region"] = str(region)
        history["unit"] = str(unit)
        daily.append(history.rename_axis("date").reset_index())
    return pd.concat(daily, ignore_index=True).sort_values(["region", "unit", "date"]).reset_index(drop=True)


def _supervised_rows(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    pieces = []
    for (region, unit), history in panel.groupby(["region", "unit"]):
        history = history.sort_values("date").set_index("date").copy()
        price = history["price"]
        stock = history["stock"]
        history["price_lag_1"] = price.shift(1)
        history["price_lag_7"] = price.shift(7)
        history["price_lag_30"] = price.shift(30)
        history["price_mean_7"] = price.shift(1).rolling(7, min_periods=7).mean()
        history["price_mean_30"] = price.shift(1).rolling(30, min_periods=30).mean()
        history["stock_ratio_30"] = stock.shift(1) / stock.shift(1).rolling(30, min_periods=7).mean()
        history["price_change_7"] = price.shift(1) / price.shift(8) - 1
        history["stock_change_7"] = stock.shift(1) / stock.shift(8) - 1
        month = history.index.month
        history["month_sin"] = np.sin(2 * np.pi * month / 12)
        history["month_cos"] = np.cos(2 * np.pi * month / 12)
        history["target_date"] = history.index + pd.to_timedelta(horizon, unit="D")
        target = history[["price"]].rename(columns={"price": "future_price"})
        history = history.join(target, on="target_date")
        history["target"] = history["future_price"] / history["price"]
        history["region"] = region
        history["unit"] = unit
        pieces.append(history.reset_index(names="date"))
    if not pieces:
        return pd.DataFrame()
    rows = pd.concat(pieces, ignore_index=True)
    rows = rows.replace([np.inf, -np.inf], np.nan)
    return rows.dropna(subset=[*FEATURES, "target"]).reset_index(drop=True)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),
        "R2": round(float(r2_score(actual, predicted)), 3) if len(actual) > 1 else float("nan"),
    }


def forecast_product(
    observations: pd.DataFrame, horizon: int, min_samples: int = 30
) -> dict[str, Any]:
    """Evaluate models on a time-ordered holdout, then forecast using all history."""
    if horizon not in (7, 30):
        raise ValueError("Forecast horizon must be 7 or 30 days.")
    panel = _daily_panel(observations)
    supervised = _supervised_rows(panel, horizon) if not panel.empty else pd.DataFrame()
    minimum = max(min_samples, horizon * 2)
    if len(supervised) < minimum:
        return {
            "status": "insufficient_data",
            "reason": f"Need at least {minimum} complete, exact-date training examples; found {len(supervised)}.",
            "samples": len(supervised),
            "metrics": {},
            "forecast": pd.DataFrame(),
            "predictions": pd.DataFrame(),
        }

    unique_dates = sorted(supervised["date"].unique())
    split_index = max(1, int(len(unique_dates) * 0.8))
    if split_index >= len(unique_dates):
        return {
            "status": "insufficient_data",
            "reason": "Not enough distinct dates for chronological train/test validation.",
            "samples": len(supervised),
            "metrics": {},
            "forecast": pd.DataFrame(),
            "predictions": pd.DataFrame(),
        }
    test_start = pd.Timestamp(unique_dates[split_index])
    train_mask = supervised["target_date"] < test_start
    test_mask = supervised["date"] >= test_start
    train, test = supervised.loc[train_mask], supervised.loc[test_mask]
    if len(train) < 10 or len(test) < 10:
        return {
            "status": "insufficient_data",
            "reason": "Chronological split produced fewer than 10 train or test examples.",
            "samples": len(supervised),
            "metrics": {},
            "forecast": pd.DataFrame(),
            "predictions": pd.DataFrame(),
        }

    regions = sorted(supervised["region"].astype(str).unique())

    def design(data: pd.DataFrame) -> pd.DataFrame:
        categorical = pd.get_dummies(data["region"].astype(str), prefix="region", dtype=float)
        result = pd.concat([data[list(FEATURES)].reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
        return result.reindex(columns=[*FEATURES, *[f"region_{name}" for name in regions]], fill_value=0)

    X_train, X_test = design(train), design(test)
    actual = test["future_price"].to_numpy(dtype=float)
    current_prices = test["price"].to_numpy(dtype=float)
    metrics: dict[str, dict[str, float]] = {
        "Persistence baseline": _metrics(actual, current_prices)
    }
    fitted: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for name, factory in MODEL_CLASSES.items():
        estimator = factory()
        estimator.fit(X_train, train["target"].to_numpy(dtype=float))
        predicted = estimator.predict(X_test) * current_prices
        fitted[name] = estimator
        test_predictions[name] = predicted
        metrics[name] = _metrics(actual, predicted)
    winner = min(MODEL_CLASSES, key=lambda model: metrics[model]["MAE"])
    model = MODEL_CLASSES[winner]()
    model.fit(design(supervised), supervised["target"].to_numpy(dtype=float))

    latest_rows = panel.sort_values("date").groupby(["region", "unit"]).tail(1).copy()
    latest_features = _supervised_rows_for_latest(panel, latest_rows)
    if latest_features.empty:
        return {
            "status": "insufficient_data",
            "reason": "The latest observations do not have enough past history for a forecast.",
            "samples": len(supervised),
            "metrics": metrics,
            "forecast": pd.DataFrame(),
            "predictions": pd.DataFrame(),
        }
    X_latest = design(latest_features)
    ratios = model.predict(X_latest)
    point = latest_features["price"].to_numpy(dtype=float) * ratios
    validation_residuals = actual - test_predictions[winner]
    low_residual, high_residual = np.quantile(validation_residuals, [0.05, 0.95])
    forecast = latest_features[["region", "unit", "date", "price"]].copy()
    forecast["predicted_price"] = np.maximum(0, point).round(2)
    forecast["lower_90"] = np.maximum(0, point + low_residual).round(2)
    forecast["upper_90"] = np.maximum(0, point + high_residual).round(2)
    forecast["target_date"] = forecast["date"] + pd.Timedelta(days=horizon)
    forecast["model"] = winner
    predictions = test[["date", "target_date", "region", "unit", "future_price"]].copy()
    predictions["actual_price"] = actual
    predictions["predicted_price"] = test_predictions[winner]
    return {
        "status": "ok",
        "reason": None,
        "samples": len(supervised),
        "train_samples": len(train),
        "test_samples": len(test),
        "split_date": test_start,
        "training_target_end": train["target_date"].max(),
        "model": winner,
        "metrics": metrics,
        "forecast": forecast.reset_index(drop=True),
        "predictions": predictions.reset_index(drop=True),
        "uncertainty": "90% empirical test-residual interval",
    }


def _supervised_rows_for_latest(panel: pd.DataFrame, latest_rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for row in latest_rows.itertuples(index=False):
        region_history = panel[
            (panel["region"] == row.region) & (panel["unit"] == row.unit)
        ].sort_values("date")
        prices = region_history["price"].to_numpy(dtype=float)
        stocks = region_history["stock"].to_numpy(dtype=float)
        if len(prices) < 31:
            continue
        stock_recent = stocks[-31:-1]
        stock_mean = np.nanmean(stock_recent)
        record = {
            "region": str(row.region),
            "unit": str(row.unit),
            "date": pd.Timestamp(row.date),
            "price": float(row.price),
            "price_lag_1": prices[-2],
            "price_lag_7": prices[-8],
            "price_lag_30": prices[-31],
            "price_mean_7": float(np.mean(prices[-8:-1])),
            "price_mean_30": float(np.mean(prices[-31:-1])),
            "stock_ratio_30": float(stocks[-2] / stock_mean) if _finite(stock_mean) and stock_mean else np.nan,
            "price_change_7": float(prices[-2] / prices[-9] - 1) if prices[-9] else np.nan,
            "stock_change_7": float(stocks[-2] / stocks[-9] - 1) if _finite(stocks[-9]) and stocks[-9] else np.nan,
            "month_sin": float(np.sin(2 * np.pi * pd.Timestamp(row.date).month / 12)),
            "month_cos": float(np.cos(2 * np.pi * pd.Timestamp(row.date).month / 12)),
        }
        if all(_finite(record[column]) for column in FEATURES):
            result.append(record)
    return pd.DataFrame(result)


def detect_shortage_risks(observations: pd.DataFrame) -> pd.DataFrame:
    """Emit explainable watch/anomaly signals; never infer a confirmed shortage."""
    columns = [
        "product", "region", "unit", "date", "risk_level", "status", "reason",
        "price_change_pct", "stock_ratio", "data_quality", "recommended_action",
    ]
    if observations.empty:
        return pd.DataFrame(columns=columns)
    frame = observations.copy()
    if "unit" not in frame:
        frame["unit"] = ""
    frame["unit"] = frame["unit"].fillna("").astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["price"] = pd.to_numeric(frame.get("price"), errors="coerce")
    quantity_column = "quantity" if "quantity" in frame else "available_quantity"
    if quantity_column not in frame:
        frame[quantity_column] = np.nan
    frame[quantity_column] = pd.to_numeric(frame[quantity_column], errors="coerce")
    output = []
    for (product, region, unit), group in frame.dropna(subset=["date"]).groupby(
        ["product", "region", "unit"]
    ):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        price_history = group.loc[group["price"].notna(), "price"].tail(30).iloc[:-1]
        stock_history = group.loc[group[quantity_column].notna(), quantity_column].tail(30).iloc[:-1]
        price_change = (
            float(latest["price"] / price_history.median() - 1)
            if _finite(latest.get("price")) and not price_history.empty and price_history.median() > 0
            else np.nan
        )
        stock_ratio = (
            float(latest[quantity_column] / stock_history.median())
            if _finite(latest.get(quantity_column)) and not stock_history.empty and stock_history.median() > 0
            else np.nan
        )
        reasons = []
        score = 0
        if _finite(price_change) and price_change >= 0.2:
            score += 2
            reasons.append(f"Price is {price_change * 100:.1f}% above the preceding local median.")
        elif _finite(price_change) and price_change >= 0.1:
            score += 1
            reasons.append(f"Price is {price_change * 100:.1f}% above the preceding local median.")
        if _finite(stock_ratio) and stock_ratio <= 0.5:
            score += 2
            reasons.append(f"Reported stock is {stock_ratio:.2f}× the preceding local median.")
        elif _finite(stock_ratio) and stock_ratio <= 0.75:
            score += 1
            reasons.append(f"Reported stock is {stock_ratio:.2f}× the preceding local median.")
        if score == 0:
            continue
        level = "RED" if score >= 4 else "ORANGE" if score >= 3 else "YELLOW"
        # Price-only spikes are anomalies, not evidence that a shortage exists.
        status = "ANOMALY" if _finite(price_change) and price_change >= 0.1 and not _finite(stock_ratio) else "POTENTIAL_RISK"
        quality = str(latest.get("data_quality", "UNVERIFIED"))
        output.append(
            {
                "product": product,
                "region": region,
                "unit": unit,
                "date": latest["date"].strftime("%Y-%m-%d"),
                "risk_level": level,
                "status": status,
                "reason": " ".join(reasons),
                "price_change_pct": round(price_change * 100, 2) if _finite(price_change) else np.nan,
                "stock_ratio": round(stock_ratio, 3) if _finite(stock_ratio) else np.nan,
                "data_quality": quality,
                "recommended_action": (
                    "Verify with market operators and official supply/stock records before escalation."
                ),
            }
        )
    return pd.DataFrame(output, columns=columns).sort_values(
        ["risk_level", "date"], ascending=[False, False], ignore_index=True
    ) if output else pd.DataFrame(columns=columns)


def simulate_scenario(
    current_price: float | None,
    current_stock: float | None,
    production_change_pct: float = 0,
    demand_change_pct: float = 0,
    distribution_delay_days: int = 0,
    stock_change_pct: float = 0,
    transport_cost_change_pct: float = 0,
) -> dict[str, Any]:
    """A transparent sensitivity calculation, not a causal or market-price forecast."""
    inputs = (production_change_pct, demand_change_pct, stock_change_pct, transport_cost_change_pct)
    if any(not _finite(value) or value < -100 for value in inputs):
        raise ValueError("Scenario percentage changes must be finite and at least -100%.")
    if distribution_delay_days < 0:
        raise ValueError("Distribution delay must be zero or greater.")
    supply_factor = max(0.0, 1 + production_change_pct / 100) * max(0.0, 1 + stock_change_pct / 100)
    demand_factor = max(0.0, 1 + demand_change_pct / 100)
    # Explicit, conservative sensitivity coefficients; not fitted causal effects.
    price_index = (
        (demand_factor / max(supply_factor, 0.05)) ** 0.35
        * (1 + transport_cost_change_pct / 100 * 0.15)
        * (1 + min(distribution_delay_days, 30) / 100)
    )
    pressure_signals = int(supply_factor <= 0.75) + int(price_index >= 1.2) + int(distribution_delay_days >= 7)
    if supply_factor <= 0.5 and price_index >= 1.2:
        hypothetical_risk = "CRITICAL"
    elif pressure_signals >= 2:
        hypothetical_risk = "HIGH"
    elif pressure_signals == 1 or supply_factor < 0.9 or price_index > 1.05:
        hypothetical_risk = "OBSERVE"
    else:
        hypothetical_risk = "NORMAL"
    return {
        "label": "Hypothetical sensitivity scenario — not a real prediction or causal estimate",
        "current_price": current_price,
        "scenario_price": round(current_price * price_index, 2) if _finite(current_price) else None,
        "current_stock": current_stock,
        "scenario_stock_index_pct": round((supply_factor - 1) * 100, 2),
        "price_index_pct": round((price_index - 1) * 100, 2),
        "hypothetical_risk": hypothetical_risk,
        "assumptions": {
            "production_change_pct": production_change_pct,
            "demand_change_pct": demand_change_pct,
            "distribution_delay_days": distribution_delay_days,
            "stock_change_pct": stock_change_pct,
            "transport_cost_change_pct": transport_cost_change_pct,
        },
    }
