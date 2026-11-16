"""
AGRI-SMART DZ - Data processing & feature engineering.

Steps: load from SQLite -> remove duplicates -> fix missing values ->
merge price + stock -> build features -> chronological train/test split.
"""
import numpy as np
import pandas as pd

if __package__:
    from . import database as db
else:
    import database as db

HORIZONS = (7, 30)  # prediction horizons in days
FEATURES_NUM = [
    "price_lag7", "price_lag30", "price_ma7", "price_ma30",
    "price_pct7", "stock_pct7", "stock_ratio",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
]


def load_clean(product="Potato"):
    """Return one clean DataFrame: date, region, price, stock, data_source."""
    prices = db.get_prices(product).drop_duplicates(subset=["date", "region"])
    stock = db.get_stock(product).drop_duplicates(subset=["date", "region"])

    df = prices.merge(
        stock[["date", "region", "available_quantity"]], on=["date", "region"], how="outer"
    ).sort_values(["region", "date"]).reset_index(drop=True)
    df = df.rename(columns={"available_quantity": "stock"})

    # Missing values: time interpolation per region, then forward/back fill edges
    for col in ("price", "stock"):
        df[col] = df.groupby("region")[col].transform(lambda s: s.interpolate(limit_direction="both"))
    # Source label for rows that came from price table (fallback if missing)
    df["data_source"] = df["data_source"].fillna("SYNTHETIC_DEMO")
    df["injected_event"] = df["injected_event"].fillna(0).astype(int)
    return df


def add_features(df):
    """Create lag / rolling / seasonal features (only uses PAST information)."""
    df = df.sort_values(["region", "date"]).copy()
    g = df.groupby("region")
    df["price_lag7"] = g["price"].shift(7) / df["price"]
    df["price_lag30"] = g["price"].shift(30) / df["price"]
    df["price_ma7"] = g["price"].transform(lambda s: s.rolling(7).mean()) / df["price"]
    df["price_ma30"] = g["price"].transform(lambda s: s.rolling(30).mean()) / df["price"]
    df["price_pct7"] = g["price"].pct_change(7)
    df["stock_pct7"] = g["stock"].pct_change(7)
    df["stock_ma30"] = g["stock"].transform(lambda s: s.rolling(30).mean())
    df["stock_ratio"] = df["stock"] / df["stock_ma30"]  # <1 means stock below recent norm
    m = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    return df


def build_supervised(df, horizon):
    """Add target = price `horizon` days ahead, one-hot regions, drop incomplete rows."""
    df = add_features(df)
    # Target = future price / today's price (a ratio). Predicting the ratio instead of the raw
    # price makes the model robust to long-term price inflation (trees cannot extrapolate).
    df["future_price"] = df.groupby("region")["price"].shift(-horizon)
    df["target"] = df["future_price"] / df["price"]
    df = df.dropna(subset=FEATURES_NUM + ["target"]).reset_index(drop=True)
    X = pd.concat([df[FEATURES_NUM], pd.get_dummies(df["region"], prefix="reg", dtype=float)], axis=1)
    return df, X, df["target"]


def chrono_split(df, X, y, split_date="2026-01-01", horizon=0):
    """Split chronologically without allowing training targets to cross into the test period."""
    cut = pd.Timestamp(split_date)
    tr = df["date"] + pd.to_timedelta(horizon, unit="D") < cut
    te = df["date"] >= cut
    return X[tr], X[te], y[tr], y[te], df[tr], df[te]


if __name__ == "__main__":
    d = load_clean()
    print(d.shape, "missing:", int(d[["price", "stock"]].isna().sum().sum()))
    print(d.describe().round(2))
