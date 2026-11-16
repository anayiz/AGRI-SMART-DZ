"""
AGRI-SMART DZ - SQLite database layer.

Tables: products, prices, stock, alerts, predictions
Every prices/stock row carries a `data_source` column:
    OFFICIAL        -> verified real-world data
    MANUAL          -> entered by a user in the app
    SYNTHETIC_DEMO  -> simulated demonstration data
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "agri_smart.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL UNIQUE,
    category     TEXT,
    unit         TEXT
);
CREATE TABLE IF NOT EXISTS prices (
    price_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id     INTEGER NOT NULL REFERENCES products(product_id),
    region         TEXT NOT NULL,
    date           TEXT NOT NULL,
    price          REAL,
    data_source    TEXT NOT NULL DEFAULT 'SYNTHETIC_DEMO',
    injected_event INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS stock (
    stock_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id         INTEGER NOT NULL REFERENCES products(product_id),
    region             TEXT NOT NULL,
    date               TEXT NOT NULL,
    available_quantity REAL,
    data_source        TEXT NOT NULL DEFAULT 'SYNTHETIC_DEMO',
    injected_event     INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type  TEXT NOT NULL,
    severity    TEXT NOT NULL,
    region      TEXT,
    description TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    region          TEXT,
    predicted_price REAL,
    prediction_date TEXT,
    model_version   TEXT
);
CREATE INDEX IF NOT EXISTS idx_prices ON prices(product_id, region, date);
CREATE INDEX IF NOT EXISTS idx_stock  ON stock(product_id, region, date);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.executescript(SCHEMA)


def add_product(name, category="Vegetable", unit="DZD/kg"):
    with get_conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO products(product_name, category, unit) VALUES (?,?,?)",
            (name, category, unit),
        )
        return c.execute("SELECT product_id FROM products WHERE product_name=?", (name,)).fetchone()[0]


def get_product_id(name="Potato"):
    with get_conn() as c:
        row = c.execute("SELECT product_id FROM products WHERE product_name=?", (name,)).fetchone()
    return row[0] if row else None


def has_market_data(product_name="Potato"):
    pid = get_product_id(product_name)
    if pid is None:
        return False
    with get_conn() as c:
        prices_count, stock_count = c.execute(
            """SELECT
                   (SELECT COUNT(*) FROM prices WHERE product_id=?),
                   (SELECT COUNT(*) FROM stock WHERE product_id=?)""",
            (pid, pid),
        ).fetchone()
    return prices_count > 0 and stock_count > 0


def load_csv_into_db(product_name="Potato", prices_csv=None, stock_csv=None, reset=False):
    """Import the cleaned CSVs. Duplicates are dropped; missing prices are kept as NULL
    (they are interpolated later in data_processing.py)."""
    prices_csv = prices_csv or os.path.join(ROOT, "data", "potato_prices.csv")
    stock_csv = stock_csv or os.path.join(ROOT, "data", "potato_stock.csv")
    init_db()
    pid = add_product(product_name)
    p = pd.read_csv(prices_csv).drop_duplicates(subset=["date", "region"])
    s = pd.read_csv(stock_csv).drop_duplicates(subset=["date", "region"])
    p["product_id"] = pid
    s["product_id"] = pid

    with get_conn() as c:
        if reset:
            c.execute("DELETE FROM prices WHERE product_id=?", (pid,))
            c.execute("DELETE FROM stock WHERE product_id=?", (pid,))
        else:
            manual_prices = set(c.execute(
                "SELECT region, date FROM prices WHERE product_id=? AND data_source='MANUAL'",
                (pid,),
            ).fetchall())
            manual_stock = set(c.execute(
                "SELECT region, date FROM stock WHERE product_id=? AND data_source='MANUAL'",
                (pid,),
            ).fetchall())
            c.execute("DELETE FROM prices WHERE product_id=? AND data_source!='MANUAL'", (pid,))
            c.execute("DELETE FROM stock WHERE product_id=? AND data_source!='MANUAL'", (pid,))
            p = p.loc[
                [(region, date) not in manual_prices for region, date in zip(p["region"], p["date"])]
            ]
            s = s.loc[
                [(region, date) not in manual_stock for region, date in zip(s["region"], s["date"])]
            ]
        p[["product_id", "region", "date", "price", "data_source", "injected_event"]].to_sql(
            "prices", c, if_exists="append", index=False)
        s[["product_id", "region", "date", "available_quantity", "data_source", "injected_event"]].to_sql(
            "stock", c, if_exists="append", index=False)
    return len(p), len(s)


# ---------- read helpers ----------
def get_prices(product_name="Potato"):
    with get_conn() as c:
        return pd.read_sql_query(
            """SELECT pr.date, pr.region, pr.price, pr.data_source, pr.injected_event
               FROM prices pr JOIN products p ON p.product_id = pr.product_id
               WHERE p.product_name = ? ORDER BY pr.date, pr.region""",
            c, params=(product_name,), parse_dates=["date"])


def get_stock(product_name="Potato"):
    with get_conn() as c:
        return pd.read_sql_query(
            """SELECT s.date, s.region, s.available_quantity, s.data_source, s.injected_event
               FROM stock s JOIN products p ON p.product_id = s.product_id
               WHERE p.product_name = ? ORDER BY s.date, s.region""",
            c, params=(product_name,), parse_dates=["date"])


# ---------- write helpers ----------
def add_manual_stock(product_name, region, date, quantity):
    pid = get_product_id(product_name)
    with get_conn() as c:
        c.execute(
            "DELETE FROM stock WHERE product_id=? AND region=? AND date=?",
            (pid, region, str(date)),
        )
        c.execute(
            "INSERT INTO stock(product_id, region, date, available_quantity, data_source) VALUES (?,?,?,?, 'MANUAL')",
            (pid, region, str(date), float(quantity)))


def add_manual_price(product_name, region, date, price):
    pid = get_product_id(product_name)
    with get_conn() as c:
        c.execute(
            "DELETE FROM prices WHERE product_id=? AND region=? AND date=?",
            (pid, region, str(date)),
        )
        c.execute(
            "INSERT INTO prices(product_id, region, date, price, data_source) VALUES (?,?,?,?, 'MANUAL')",
            (pid, region, str(date), float(price)))


def add_manual_observation(product_name, region, date, price, quantity):
    pid = get_product_id(product_name)
    if pid is None:
        raise ValueError(f"Unknown product: {product_name}")
    if float(price) <= 0 or float(quantity) < 0:
        raise ValueError("Price must be positive and stock cannot be negative.")
    with get_conn() as c:
        c.execute(
            "DELETE FROM prices WHERE product_id=? AND region=? AND date=?",
            (pid, region, str(date)),
        )
        c.execute(
            "DELETE FROM stock WHERE product_id=? AND region=? AND date=?",
            (pid, region, str(date)),
        )
        c.execute(
            "INSERT INTO prices(product_id, region, date, price, data_source) VALUES (?,?,?,?, 'MANUAL')",
            (pid, region, str(date), float(price)),
        )
        c.execute(
            "INSERT INTO stock(product_id, region, date, available_quantity, data_source) VALUES (?,?,?,?, 'MANUAL')",
            (pid, region, str(date), float(quantity)),
        )


def save_alert(alert_type, severity, region, description, created_at=None):
    created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as c:
        c.execute(
            "INSERT INTO alerts(alert_type, severity, region, description, created_at) VALUES (?,?,?,?,?)",
            (alert_type, severity, region, description, created_at))


def get_alerts(limit=100):
    with get_conn() as c:
        return pd.read_sql_query(
            "SELECT * FROM alerts ORDER BY created_at DESC, alert_id DESC LIMIT ?", c, params=(limit,))


def save_prediction(product_name, region, predicted_price, prediction_date, model_version):
    pid = get_product_id(product_name)
    with get_conn() as c:
        c.execute(
            "INSERT INTO predictions(product_id, region, predicted_price, prediction_date, model_version) VALUES (?,?,?,?,?)",
            (pid, region, float(predicted_price), str(prediction_date), model_version))


def get_predictions(limit=200):
    with get_conn() as c:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT ?", c, params=(limit,))


def last_update(table="prices"):
    with get_conn() as c:
        row = c.execute(f"SELECT MAX(date) FROM {table}").fetchone()
    return row[0]


if __name__ == "__main__":
    n_p, n_s = load_csv_into_db(reset=False)
    print(f"Database ready at {DB_PATH}: {n_p} price rows, {n_s} stock rows loaded.")
