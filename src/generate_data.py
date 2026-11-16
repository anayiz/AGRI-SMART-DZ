"""
AGRI-SMART DZ - Synthetic demonstration data generator (POTATOES).

!!! IMPORTANT !!!
Everything produced by this script is SYNTHETIC DEMONSTRATION DATA.
It is NOT real Algerian market statistics. Every row carries
data_source = "SYNTHETIC_DEMO" so the app can label it everywhere.

When real data (e.g. from official sources) becomes available, replace
the two CSV files in /data using the same columns and set
data_source = "OFFICIAL" (or "MANUAL" for hand-entered rows).

Run from the project root:   python src/generate_data.py
"""
import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

START, END = "2023-01-01", "2026-10-31"
# Region -> (price multiplier, production capacity weight)
REGIONS = {
    "Alger":        (1.10, 0.6),
    "Oran":         (1.05, 0.7),
    "Constantine":  (1.03, 0.8),
    "Mostaganem":   (0.95, 1.4),
    "Ain Defla":    (0.92, 1.6),
    "El Oued":      (0.90, 1.7),
    "Setif":        (1.00, 1.0),
    "Tizi Ouzou":   (1.08, 0.5),
}
BASE_PRICE = 55.0  # DZD/kg, arbitrary synthetic baseline


def seasonal_price_factor(doy: np.ndarray) -> np.ndarray:
    # Synthetic pattern: higher prices in late summer/autumn (inter-season gap)
    return 1.0 + 0.18 * np.sin(2 * np.pi * (doy - 150) / 365)


def seasonal_supply_factor(doy: np.ndarray) -> np.ndarray:
    # Synthetic pattern: supply is the mirror of price (more supply -> lower price)
    return 1.0 - 0.22 * np.sin(2 * np.pi * (doy - 150) / 365)


def main():
    dates = pd.date_range(START, END, freq="D")
    doy = dates.dayofyear.to_numpy()
    n = len(dates)

    # Slow inflation-like trend (synthetic)
    trend = np.linspace(1.0, 1.25, n)

    price_rows, stock_rows = [], []

    # Injected events (hidden ground truth, useful to TEST the detectors)
    # (region, start_date, length_days, price_mult, stock_mult)
    events = [
        ("Alger",       "2024-08-10", 12, 1.45, 0.55),
        ("Oran",        "2025-09-01", 15, 1.40, 0.50),
        ("Constantine", "2026-07-20", 10, 1.50, 0.45),
        ("Tizi Ouzou",  "2025-03-05",  8, 1.35, 0.60),
        ("Setif",       "2026-09-15", 14, 1.42, 0.52),
        # ---- DEMO SCENARIO "Potato Market Pressure - July 2026" (synthetic) ----
        ("Alger",       "2026-07-10", 16, 1.35, 1.00),  # price rise only
        ("Oran",        "2026-07-12", 14, 1.00, 0.55),  # stock decline only
        ("Constantine", "2026-07-20", 10, 1.50, 0.45),  # price up AND stock down
    ]
    flag = {r: np.zeros(n, dtype=int) for r in REGIONS}
    price_ev = {r: np.ones(n) for r in REGIONS}
    stock_ev = {r: np.ones(n) for r in REGIONS}
    for reg, s, L, pm, sm in events:
        i0 = int(np.searchsorted(dates, pd.Timestamp(s)))
        sl = slice(i0, min(i0 + L, n))
        price_ev[reg][sl] = pm
        stock_ev[reg][sl] = sm
        flag[reg][sl] = 1

    for reg, (pmult, cap) in REGIONS.items():
        noise_p = rng.normal(0, 0.025, n)
        # autocorrelated noise (prices are sticky day to day)
        for i in range(1, n):
            noise_p[i] = 0.7 * noise_p[i - 1] + noise_p[i]
        price = BASE_PRICE * pmult * trend * seasonal_price_factor(doy) * (1 + noise_p) * price_ev[reg]
        price = np.clip(price, 15, None).round(2)

        base_stock = 400.0 * cap  # tonnes, arbitrary synthetic scale
        noise_s = rng.normal(0, 0.05, n)
        stock = base_stock * seasonal_supply_factor(doy) * (1 + noise_s) * stock_ev[reg]
        stock = np.clip(stock, 5, None).round(1)

        for d, p, s, f in zip(dates, price, stock, flag[reg]):
            price_rows.append((d.strftime("%Y-%m-%d"), reg, float(p), "SYNTHETIC_DEMO", int(f)))
            stock_rows.append((d.strftime("%Y-%m-%d"), reg, float(s), "SYNTHETIC_DEMO", int(f)))

    prices = pd.DataFrame(price_rows, columns=["date", "region", "price", "data_source", "injected_event"])
    stock = pd.DataFrame(stock_rows, columns=["date", "region", "available_quantity", "data_source", "injected_event"])

    # Simulate a few real-world data problems so preprocessing has work to do
    miss = prices.sample(frac=0.01, random_state=SEED).index
    prices.loc[miss, "price"] = np.nan
    dup = prices.sample(n=20, random_state=SEED)
    prices = pd.concat([prices, dup], ignore_index=True)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    prices.to_csv(os.path.join(root, "data", "potato_prices.csv"), index=False)
    stock.to_csv(os.path.join(root, "data", "potato_stock.csv"), index=False)
    print(f"potato_prices.csv: {len(prices)} rows | potato_stock.csv: {len(stock)} rows")
    print("REMINDER: all data is SYNTHETIC_DEMO - never present it as real Algerian statistics.")


if __name__ == "__main__":
    main()
