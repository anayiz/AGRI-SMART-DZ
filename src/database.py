"""Persistence layer for AGRI-SMART DZ.

SQLite remains the zero-configuration local/demo backend. Set DATABASE_URL to a
PostgreSQL URL for a persistent hosted database.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    URL,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import NullPool

DB_PATH = str(Path(__file__).resolve().parents[1] / "agri_smart.db")

QUALITY_LEVELS = ("OFFICIAL", "VERIFIED", "UNVERIFIED", "SYNTHETIC_DEMO")
_engine: Engine | None = None
_engine_key: str | None = None
_initialized_key: str | None = None
_init_lock = threading.RLock()
_metadata = MetaData()

categories = Table(
    "product_categories",
    _metadata,
    Column("category_id", Integer, primary_key=True, autoincrement=True),
    Column("category_name", String(80), nullable=False, unique=True),
)
products = Table(
    "products",
    _metadata,
    Column("product_id", Integer, primary_key=True, autoincrement=True),
    Column("product_name", String(120), nullable=False, unique=True),
    Column("category_id", Integer, ForeignKey("product_categories.category_id")),
    Column("unit", String(40), nullable=False, server_default="kg"),
    Column("is_active", Boolean, nullable=False, server_default=text("TRUE")),
    Column("created_at", DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)
wilayas = Table(
    "wilayas",
    _metadata,
    Column("wilaya_code", String(2), primary_key=True),
    Column("name_fr", String(120), nullable=False),
    Column("name_ar", String(120)),
    Column("name_en", String(120)),
)
data_sources = Table(
    "data_sources",
    _metadata,
    Column("source_id", Integer, primary_key=True, autoincrement=True),
    Column("source_name", String(200), nullable=False, unique=True),
    Column("source_type", String(40), nullable=False, server_default="MANUAL"),
    Column("source_url", Text),
    Column("notes", Text),
    Column("is_active", Boolean, nullable=False, server_default=text("TRUE")),
)
market_observations = Table(
    "market_observations",
    _metadata,
    Column("observation_id", Integer, primary_key=True, autoincrement=True),
    Column("product_id", Integer, ForeignKey("products.product_id"), nullable=False),
    Column("wilaya_code", String(2), ForeignKey("wilayas.wilaya_code")),
    Column("region", String(120), nullable=False),
    Column("market", String(160), nullable=False, server_default=""),
    Column("price", Float),
    Column("quantity", Float),
    Column("incoming_supply", Float),
    Column("unit", String(40), nullable=False, server_default="kg"),
    Column("observation_date", Date, nullable=False),
    Column("source_id", Integer, ForeignKey("data_sources.source_id"), nullable=False),
    Column("data_quality", String(24), nullable=False),
    Column("collection_method", String(40), nullable=False),
    Column("notes", Text),
    Column("created_at", DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint("price IS NULL OR price >= 0", name="ck_observation_price_nonnegative"),
    CheckConstraint("quantity IS NULL OR quantity >= 0", name="ck_observation_quantity_nonnegative"),
    CheckConstraint(
        "incoming_supply IS NULL OR incoming_supply >= 0",
        name="ck_observation_supply_nonnegative",
    ),
    CheckConstraint(
        "price IS NOT NULL OR quantity IS NOT NULL OR incoming_supply IS NOT NULL",
        name="ck_observation_has_measurement",
    ),
    CheckConstraint(
        "data_quality IN ('OFFICIAL','VERIFIED','UNVERIFIED','SYNTHETIC_DEMO')",
        name="ck_observation_quality",
    ),
    UniqueConstraint(
        "product_id",
        "region",
        "market",
        "observation_date",
        "source_id",
        name="uq_observation_natural_key",
    ),
)
Index("ix_observations_product_date", market_observations.c.product_id, market_observations.c.observation_date)
Index("ix_observations_region_date", market_observations.c.wilaya_code, market_observations.c.observation_date)
alerts = Table(
    "alerts",
    _metadata,
    Column("alert_id", Integer, primary_key=True, autoincrement=True),
    Column("product_id", Integer, ForeignKey("products.product_id")),
    Column("region", String(120), nullable=False),
    Column("wilaya_code", String(2), ForeignKey("wilayas.wilaya_code")),
    Column("alert_type", String(60), nullable=False),
    Column("risk_level", String(24), nullable=False),
    Column("unit", String(40), nullable=False, server_default=""),
    Column("description", Text, nullable=False),
    Column("indicators_json", Text),
    Column("recommended_action", Text),
    Column("status", String(32), nullable=False, server_default="POTENTIAL_RISK"),
    Column("detected_at", DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column("data_quality", String(24), nullable=False, server_default="UNVERIFIED"),
)
predictions = Table(
    "predictions",
    _metadata,
    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
    Column("product_id", Integer, ForeignKey("products.product_id")),
    Column("region", String(120), nullable=False, server_default="National"),
    Column("model_name", String(80), nullable=False),
    Column("horizon_days", Integer, nullable=False),
    Column("target_date", Date),
    Column("predicted_price", Float, nullable=False),
    Column("unit", String(40), nullable=False, server_default="kg"),
    Column("lower_bound", Float),
    Column("upper_bound", Float),
    Column("evaluation_json", Text),
    Column("created_at", DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)
scenario_runs = Table(
    "scenario_runs",
    _metadata,
    Column("scenario_id", Integer, primary_key=True, autoincrement=True),
    Column("scenario_type", String(60), nullable=False),
    Column("factor", Float, nullable=False),
    Column("assumptions_json", Text, nullable=False),
    Column("results_json", Text, nullable=False),
    Column("created_at", DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)

_CATEGORY_PRODUCTS = {
    "Vegetables": ("Potato", "Tomato", "Onion", "Garlic", "Carrot"),
    "Oils": ("Cooking Oil",),
    "Fruits": ("Apple", "Orange", "Banana"),
    "Cereals": ("Wheat", "Flour", "Semolina"),
    "Essential Food Products": ("Milk", "Sugar", "Rice"),
}


def _database_url() -> str | URL:
    configured = os.environ.get("DATABASE_URL")
    if configured:
        if configured.startswith("postgres://"):
            configured = "postgresql+psycopg://" + configured[len("postgres://") :]
        elif configured.startswith("postgresql://"):
            configured = "postgresql+psycopg://" + configured[len("postgresql://") :]
        return configured
    return URL.create("sqlite", database=os.path.abspath(DB_PATH))


def _get_engine() -> Engine:
    global _engine, _engine_key
    url = _database_url()
    key = str(url)
    if _engine is None or key != _engine_key:
        if _engine is not None:
            _engine.dispose()
        connect_args = {"check_same_thread": False} if str(url).startswith("sqlite") else {}
        _engine = create_engine(
            url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
            poolclass=NullPool if str(url).startswith("sqlite") else None,
        )
        if _engine.dialect.name == "sqlite":
            @event.listens_for(_engine, "connect")
            def _enable_sqlite_constraints(dbapi_connection: Any, _connection_record: Any) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()
        _engine_key = key
    return _engine


@contextmanager
def get_conn() -> Iterator[Connection]:
    """Yield a transactional connection for use by this module and tests."""
    conn = _get_engine().connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_database_backend() -> str:
    return _get_engine().dialect.name


def _get_or_create_source(
    conn: Connection,
    source_name: str,
    source_type: str,
    source_url: str | None = None,
    notes: str | None = None,
) -> int:
    row = conn.execute(
        text("SELECT source_id FROM data_sources WHERE source_name = :name"),
        {"name": source_name},
    ).first()
    if row:
        return int(row[0])
    conn.execute(
        text(
            "INSERT INTO data_sources (source_name, source_type, source_url, notes) "
            "VALUES (:name, :source_type, :url, :notes)"
        ),
        {"name": source_name, "source_type": source_type, "url": source_url, "notes": notes},
    )
    return int(
        conn.execute(
            text("SELECT source_id FROM data_sources WHERE source_name = :name"),
            {"name": source_name},
        ).scalar_one()
    )


def _product_id(conn: Connection, product_name: str, cache: dict[str, int] | None = None) -> int:
    normalized = product_name.strip()
    if not normalized:
        raise ValueError("Product name is required.")
    if cache is not None and normalized.casefold() in cache:
        return cache[normalized.casefold()]
    value = conn.execute(
        text("SELECT product_id FROM products WHERE lower(product_name) = lower(:name)"),
        {"name": normalized},
    ).scalar_one_or_none()
    if value is None:
        conn.execute(
            text("INSERT INTO products (product_name, unit) VALUES (:name, 'kg')"),
            {"name": normalized},
        )
        value = conn.execute(
            text("SELECT product_id FROM products WHERE lower(product_name) = lower(:name)"),
            {"name": normalized},
        ).scalar_one()
    product_id = int(value)
    if cache is not None:
        cache[normalized.casefold()] = product_id
    return product_id


def _quality_for_source(source_name: str, explicit: str | None = None) -> str:
    if explicit:
        quality = explicit.strip().upper()
    elif source_name.strip().upper() in ("SYNTHETIC", "SYNTHETIC_DEMO", "DEMO"):
        quality = "SYNTHETIC_DEMO"
    else:
        quality = "UNVERIFIED"
    if quality not in QUALITY_LEVELS:
        raise ValueError(f"Data quality must be one of: {', '.join(QUALITY_LEVELS)}.")
    return quality


def _as_date(value: Any) -> date:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("Observation date is invalid.")
    return parsed.date()


def _validated_measurement(value: Any, label: str) -> float | None:
    if value is None or pd.isna(value) or value == "":
        return None
    number = float(value)
    if not pd.notna(number) or number < 0:
        raise ValueError(f"{label} must be a non-negative number.")
    return number


def _upsert_observation(
    conn: Connection,
    observation: dict[str, Any],
    product_cache: dict[str, int] | None = None,
    source_cache: dict[str, int] | None = None,
) -> str:
    product_id = _product_id(conn, str(observation["product"]), product_cache)
    source_name = str(observation.get("source_name") or "MANUAL").strip()
    if not source_name:
        raise ValueError("Source name is required.")
    source_key = source_name.casefold()
    if source_cache is not None and source_key in source_cache:
        source_id = source_cache[source_key]
    else:
        source_id = _get_or_create_source(
            conn,
            source_name,
            str(observation.get("source_type") or "MANUAL").upper(),
            observation.get("source_url"),
            observation.get("source_notes"),
        )
        if source_cache is not None:
            source_cache[source_key] = source_id

    region = str(observation.get("region") or "").strip()
    if not region:
        raise ValueError("Wilaya or region is required.")
    market = str(observation.get("market") or "").strip()
    unit = str(observation.get("unit") or "kg").strip()
    if not unit:
        raise ValueError("Unit of measurement is required.")
    code = str(observation.get("wilaya_code") or "").strip() or None
    if code:
        exists = conn.execute(
            text("SELECT 1 FROM wilayas WHERE wilaya_code = :code"), {"code": code}
        ).first()
        if not exists:
            raise ValueError(f"Unknown wilaya code: {code}")
    price = _validated_measurement(observation.get("price"), "Price")
    quantity = _validated_measurement(observation.get("quantity"), "Stock quantity")
    supply = _validated_measurement(observation.get("incoming_supply"), "Incoming supply")
    if price is None and quantity is None and supply is None:
        raise ValueError("At least one of price, stock quantity, or incoming supply is required.")
    quality = _quality_for_source(source_name, observation.get("data_quality"))
    method = str(observation.get("collection_method") or "MANUAL").strip().upper()
    observed_on = _as_date(observation.get("observation_date") or observation.get("date"))
    natural_key = {
        "product_id": product_id,
        "region": region,
        "market": market,
        "observation_date": observed_on,
        "source_id": source_id,
    }
    existing_id = conn.execute(
        text(
            "SELECT observation_id FROM market_observations "
            "WHERE product_id=:product_id AND region=:region AND market=:market "
            "AND observation_date=:observation_date AND source_id=:source_id"
        ),
        natural_key,
    ).scalar_one_or_none()
    values = {
        **natural_key,
        "wilaya_code": code,
        "price": price,
        "quantity": quantity,
        "incoming_supply": supply,
        "unit": unit,
        "data_quality": quality,
        "collection_method": method,
        "notes": observation.get("notes"),
    }
    if existing_id is None:
        conn.execute(
            text(
                "INSERT INTO market_observations "
                "(product_id, wilaya_code, region, market, price, quantity, incoming_supply, "
                "unit, observation_date, source_id, data_quality, collection_method, notes) "
                "VALUES (:product_id, :wilaya_code, :region, :market, :price, :quantity, "
                ":incoming_supply, :unit, :observation_date, :source_id, :data_quality, "
                ":collection_method, :notes)"
            ),
            values,
        )
        operation = "inserted"
    else:
        conn.execute(
            text(
                "UPDATE market_observations SET wilaya_code=:wilaya_code, price=:price, "
                "quantity=:quantity, incoming_supply=:incoming_supply, unit=:unit, "
                "data_quality=:data_quality, collection_method=:collection_method, notes=:notes "
                "WHERE observation_id=:observation_id"
            ),
            {**values, "observation_id": existing_id},
        )
        operation = "updated"

    # Keep the original tables readable for older project code and exports.
    legacy_source = source_name
    if price is not None:
        conn.execute(
            text(
                "DELETE FROM prices WHERE product_id=:product_id AND region=:region "
                "AND date=:date AND data_source=:source"
            ),
            {"product_id": product_id, "region": region, "date": observed_on, "source": legacy_source},
        )
        conn.execute(
            text(
                "INSERT INTO prices (product_id, region, date, price, data_source) "
                "VALUES (:product_id, :region, :date, :price, :source)"
            ),
            {"product_id": product_id, "region": region, "date": observed_on, "price": price, "source": legacy_source},
        )
    if quantity is not None or supply is not None:
        conn.execute(
            text(
                "DELETE FROM stock WHERE product_id=:product_id AND region=:region "
                "AND date=:date AND data_source=:source"
            ),
            {"product_id": product_id, "region": region, "date": observed_on, "source": legacy_source},
        )
        conn.execute(
            text(
                "INSERT INTO stock (product_id, region, date, available_quantity, data_source) "
                "VALUES (:product_id, :region, :date, :quantity, :source)"
            ),
            {
                "product_id": product_id,
                "region": region,
                "date": observed_on,
                "quantity": quantity if quantity is not None else supply,
                "source": legacy_source,
            },
        )
    return operation


def _ensure_legacy_schema(conn: Connection) -> None:
    """Create the original tables if needed, then add safe nullable/defaulted columns."""
    conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS prices ("
            "price_id INTEGER PRIMARY KEY, product_id INTEGER NOT NULL, region TEXT NOT NULL, "
            "date DATE NOT NULL, price REAL NOT NULL, data_source TEXT DEFAULT 'SYNTHETIC_DEMO')"
        )
    )
    conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS stock ("
            "stock_id INTEGER PRIMARY KEY, product_id INTEGER NOT NULL, region TEXT NOT NULL, "
            "date DATE NOT NULL, available_quantity REAL NOT NULL, data_source TEXT DEFAULT 'SYNTHETIC_DEMO')"
        )
    )
    # The legacy tables are owned by earlier app versions. Their original columns are kept.
    existing = inspect(conn)
    for table_name, columns in (
        (
            "products",
            {
                "category_id": "INTEGER",
                "unit": "TEXT NOT NULL DEFAULT 'kg'",
                "is_active": "BOOLEAN NOT NULL DEFAULT TRUE",
                "created_at": "TIMESTAMP",
            },
        ),
        ("prices", {"data_source": "TEXT DEFAULT 'SYNTHETIC_DEMO'"}),
        ("stock", {"data_source": "TEXT DEFAULT 'SYNTHETIC_DEMO'"}),
        (
            "alerts",
            {
                "product_id": "INTEGER",
                "wilaya_code": "TEXT",
                "alert_type": "TEXT NOT NULL DEFAULT 'LEGACY'",
                "risk_level": "TEXT NOT NULL DEFAULT 'GREEN'",
                "unit": "TEXT NOT NULL DEFAULT ''",
                "description": "TEXT NOT NULL DEFAULT ''",
                "indicators_json": "TEXT",
                "recommended_action": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'POTENTIAL_RISK'",
                "detected_at": "TIMESTAMP",
                "data_quality": "TEXT NOT NULL DEFAULT 'UNVERIFIED'",
            },
        ),
        (
            "predictions",
            {
                "product_id": "INTEGER",
                "region": "TEXT NOT NULL DEFAULT 'National'",
                "model_name": "TEXT NOT NULL DEFAULT 'legacy'",
                "horizon_days": "INTEGER NOT NULL DEFAULT 7",
                "target_date": "DATE",
                "unit": "TEXT NOT NULL DEFAULT 'kg'",
                "lower_bound": "REAL",
                "upper_bound": "REAL",
                "evaluation_json": "TEXT",
            },
        ),
    ):
        present = {column["name"] for column in existing.get_columns(table_name)}
        for column_name, declaration in columns.items():
            if column_name not in present:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {declaration}"))
                if column_name == "created_at":
                    conn.execute(
                        text(
                            f"UPDATE {table_name} SET created_at=CURRENT_TIMESTAMP "
                            "WHERE created_at IS NULL"
                        )
                    )
                elif table_name == "alerts" and column_name == "detected_at":
                    columns_now = {column["name"] for column in inspect(conn).get_columns(table_name)}
                    legacy_date = "date" if "date" in columns_now else None
                    if legacy_date:
                        conn.execute(
                            text(
                                "UPDATE alerts SET detected_at=date WHERE detected_at IS NULL "
                                "AND date IS NOT NULL"
                            )
                        )
                    conn.execute(
                        text("UPDATE alerts SET detected_at=CURRENT_TIMESTAMP WHERE detected_at IS NULL")
                    )
    existing_alerts = {column["name"] for column in inspect(conn).get_columns("alerts")}
    if "reason" in existing_alerts and "description" in existing_alerts:
        conn.execute(
            text("UPDATE alerts SET description=reason WHERE description='' AND reason IS NOT NULL")
        )
    if "alert_level" in existing_alerts:
        conn.execute(
            text(
                "UPDATE alerts SET risk_level=alert_level "
                "WHERE risk_level='GREEN' AND alert_level IS NOT NULL"
            )
        )
    if "suggested_action" in existing_alerts:
        conn.execute(
            text(
                "UPDATE alerts SET recommended_action=suggested_action "
                "WHERE recommended_action IS NULL AND suggested_action IS NOT NULL"
            )
        )


def _seed_catalog(conn: Connection) -> None:
    product_cache: dict[str, int] = {}
    for category_name, names in _CATEGORY_PRODUCTS.items():
        category_id = conn.execute(
            text("SELECT category_id FROM product_categories WHERE category_name=:name"),
            {"name": category_name},
        ).scalar_one_or_none()
        if category_id is None:
            conn.execute(
                text("INSERT INTO product_categories (category_name) VALUES (:name)"),
                {"name": category_name},
            )
            category_id = conn.execute(
                text("SELECT category_id FROM product_categories WHERE category_name=:name"),
                {"name": category_name},
            ).scalar_one()
        for name in names:
            product_id = _product_id(conn, name, product_cache)
            conn.execute(
                text("UPDATE products SET category_id=:category_id WHERE product_id=:product_id"),
                {"category_id": category_id, "product_id": product_id},
            )


def _migrate_legacy_observations(conn: Connection) -> None:
    count = conn.execute(text("SELECT COUNT(*) FROM market_observations")).scalar_one()
    if count:
        return
    price_rows = conn.execute(
        text(
            "SELECT p.product_id, pr.region, pr.date, pr.price, pr.data_source, "
            "s.available_quantity, s.data_source AS stock_source "
            "FROM prices pr JOIN products p ON p.product_id=pr.product_id "
            "LEFT JOIN stock s ON s.product_id=pr.product_id AND s.region=pr.region AND s.date=pr.date"
        )
    ).mappings().all()
    product_names = {
        int(row["product_id"]): str(row["product_name"])
        for row in conn.execute(text("SELECT product_id, product_name FROM products")).mappings()
    }
    for row in price_rows:
        source = str(row["data_source"] or "SYNTHETIC_DEMO")
        _upsert_observation(
            conn,
            {
                "product": product_names[int(row["product_id"])],
                "region": row["region"],
                "date": row["date"],
                "price": row["price"],
                "quantity": row["available_quantity"],
                "source_name": source,
                "collection_method": "LEGACY_MIGRATION",
                "data_quality": _quality_for_source(source),
                "unit": "kg",
            },
        )


def seed_wilayas_from_geojson(path: str | None = None) -> int:
    """Load available boundary names/codes without creating observations or coordinates."""
    geojson_path = Path(path) if path else Path(__file__).resolve().parents[1] / "data" / "algeria_wilayas_58.geojson"
    if not geojson_path.exists():
        return 0
    with geojson_path.open(encoding="utf-8") as geojson_file:
        document = json.load(geojson_file)
    count = 0
    with get_conn() as conn:
        for feature in document.get("features", []):
            props = feature.get("properties") or {}
            code = str(props.get("CdWilaya_2019", "")).zfill(2)
            name = str(props.get("LbWilaya_2019", "")).strip().title()
            if len(code) != 2 or not name:
                continue
            conn.execute(
                text(
                    "INSERT INTO wilayas (wilaya_code, name_fr) VALUES (:code, :name) "
                    "ON CONFLICT (wilaya_code) DO UPDATE SET name_fr=excluded.name_fr"
                ),
                {"code": code, "name": name},
            )
            count += 1
    return count


@contextmanager
def _database_initialization_lock() -> Iterator[None]:
    """Serialize initialization across threads and separate Streamlit processes."""
    with _init_lock:
        url = _database_url()
        if str(url).startswith(("postgres", "postgresql")):
            yield
            return

        lock_path = Path(DB_PATH).with_name(Path(DB_PATH).name + ".init.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 30.0
        descriptor: int | None = None

        while descriptor is None:
            try:
                descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, str(os.getpid()).encode("ascii"))
            except FileExistsError:
                try:
                    stale_after = 120
                    if time.time() - lock_path.stat().st_mtime > stale_after:
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Timed out waiting for AGRI-SMART DZ database initialization."
                    )
                time.sleep(0.1)

        try:
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def init_db() -> None:
    global _initialized_key
    engine = _get_engine()
    key = str(_database_url())
    if _initialized_key == key:
        return

    with _database_initialization_lock():
        engine = _get_engine()
        key = str(_database_url())
        if _initialized_key == key:
            return

        if engine.dialect.name == "sqlite":
            with engine.connect() as conn:
                conn.execute(text("PRAGMA busy_timeout=30000"))
                conn.execute(text("PRAGMA journal_mode=WAL"))

        _metadata.create_all(engine)
        with get_conn() as conn:
            _ensure_legacy_schema(conn)
            _seed_catalog(conn)
            _migrate_legacy_observations(conn)
        seed_wilayas_from_geojson()
        _initialized_key = key


def list_products(active_only: bool = True) -> pd.DataFrame:
    condition = "WHERE p.is_active = TRUE" if active_only else ""
    with get_conn() as conn:
        return pd.read_sql_query(
            text(
                "SELECT p.product_id, p.product_name, p.category_id, c.category_name, p.unit, p.is_active "
                "FROM products p LEFT JOIN product_categories c ON c.category_id=p.category_id "
                f"{condition} ORDER BY c.category_name, p.product_name"
            ),
            conn,
        )


def add_product(product_name: str, category_name: str = "Other", unit: str = "kg") -> int:
    name = str(product_name).strip()
    category = str(category_name).strip()
    measure = str(unit).strip()
    if not name or len(name) > 120 or not category or len(category) > 80:
        raise ValueError("Product and category names are required and must be within length limits.")
    if not measure or len(measure) > 40:
        raise ValueError("A valid unit of measurement is required.")
    with get_conn() as conn:
        category_id = conn.execute(
            text("SELECT category_id FROM product_categories WHERE lower(category_name)=lower(:name)"),
            {"name": category},
        ).scalar_one_or_none()
        if category_id is None:
            conn.execute(text("INSERT INTO product_categories (category_name) VALUES (:name)"), {"name": category})
            category_id = conn.execute(
                text("SELECT category_id FROM product_categories WHERE lower(category_name)=lower(:name)"),
                {"name": category},
            ).scalar_one()
        product_id = _product_id(conn, name)
        conn.execute(
            text(
                "UPDATE products SET category_id=:category_id, unit=:unit, is_active=TRUE "
                "WHERE product_id=:product_id"
            ),
            {"category_id": category_id, "unit": measure, "product_id": product_id},
        )
        return product_id


def list_wilayas() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            text("SELECT wilaya_code, name_fr, name_ar, name_en FROM wilayas ORDER BY wilaya_code"),
            conn,
        )


def list_data_sources() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            text("SELECT source_id, source_name, source_type, source_url, notes FROM data_sources ORDER BY source_name"),
            conn,
        )


def add_market_observation(
    product: str,
    region: str,
    observation_date: Any,
    price: Any = None,
    quantity: Any = None,
    incoming_supply: Any = None,
    market: str = "",
    unit: str = "kg",
    source_name: str = "MANUAL",
    data_quality: str | None = None,
    collection_method: str = "MANUAL",
    wilaya_code: str | None = None,
    notes: str | None = None,
    source_url: str | None = None,
) -> str:
    with get_conn() as conn:
        return _upsert_observation(
            conn,
            {
                "product": product,
                "region": region,
                "observation_date": observation_date,
                "price": price,
                "quantity": quantity,
                "incoming_supply": incoming_supply,
                "market": market,
                "unit": unit,
                "source_name": source_name,
                "source_type": "API" if source_url else collection_method,
                "source_url": source_url,
                "data_quality": data_quality,
                "collection_method": collection_method,
                "wilaya_code": wilaya_code,
                "notes": notes,
            },
        )


def add_manual_observation(
    product: str,
    region: str,
    date: Any,
    price: Any = None,
    quantity: Any = None,
    market: str = "",
    unit: str = "kg",
    source_name: str = "MANUAL",
    data_quality: str = "UNVERIFIED",
    incoming_supply: Any = None,
    wilaya_code: str | None = None,
    notes: str | None = None,
) -> str:
    return add_market_observation(
        product=product,
        region=region,
        observation_date=date,
        price=price,
        quantity=quantity,
        incoming_supply=incoming_supply,
        market=market,
        unit=unit,
        source_name=source_name,
        data_quality=data_quality,
        collection_method="MANUAL",
        wilaya_code=wilaya_code,
        notes=notes,
    )


def add_manual_price(product: str, region: str, date: Any, price: Any) -> None:
    add_manual_observation(product, region, date, price=price)


def add_manual_stock(product: str, region: str, date: Any, quantity: Any) -> None:
    add_manual_observation(product, region, date, quantity=quantity)


def import_market_observations(rows: list[dict[str, Any]]) -> dict[str, int]:
    result = {"inserted": 0, "updated": 0, "skipped": 0}
    with get_conn() as conn:
        product_cache: dict[str, int] = {}
        source_cache: dict[str, int] = {}
        for item in rows:
            try:
                status = _upsert_observation(conn, item, product_cache, source_cache)
                result[status] += 1
            except (ValueError, TypeError):
                result["skipped"] += 1
    return result


def load_csv_into_db(
    product_name: str = "Potato",
    reset: bool = False,
    prices_csv: str | None = None,
    stock_csv: str | None = None,
) -> int:
    """Import the price/stock CSV pair, retaining compatibility with older callers."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    prices_path = Path(prices_csv) if prices_csv else data_dir / "potato_prices.csv"
    stock_path = Path(stock_csv) if stock_csv else data_dir / "potato_stock.csv"
    price_frame = pd.read_csv(prices_path) if prices_path.exists() else pd.DataFrame()
    stock_frame = pd.read_csv(stock_path) if stock_path.exists() else pd.DataFrame()
    if price_frame.empty and stock_frame.empty:
        return 0
    key_columns = ("date", "region")
    for frame, value_column in ((price_frame, "price"), (stock_frame, "available_quantity")):
        if not frame.empty and (not set(key_columns).issubset(frame.columns) or value_column not in frame.columns):
            raise ValueError(f"CSV must include date, region, and {value_column}.")
    if not price_frame.empty:
        price_frame = price_frame[[*key_columns, "price", *[c for c in ("data_source", "injected_event") if c in price_frame.columns]]]
        if "data_source" in price_frame.columns:
            price_frame = price_frame.rename(columns={"data_source": "price_source"})
    if not stock_frame.empty:
        stock_frame = stock_frame[[*key_columns, "available_quantity", *[c for c in ("data_source", "injected_event") if c in stock_frame.columns]]]
        if "data_source" in stock_frame.columns:
            stock_frame = stock_frame.rename(columns={"data_source": "stock_source"})
        stock_frame = stock_frame.drop_duplicates(list(key_columns), keep="last")
    merged = (
        price_frame.merge(stock_frame, on=list(key_columns), how="outer", suffixes=("_price", "_stock"))
        if not price_frame.empty and not stock_frame.empty
        else price_frame if not price_frame.empty else stock_frame
    )
    if reset:
        with get_conn() as conn:
            conn.execute(text("DELETE FROM market_observations"))
            conn.execute(text("DELETE FROM prices"))
            conn.execute(text("DELETE FROM stock"))
    rows = []
    for record in merged.to_dict(orient="records"):
        source = record.get("price_source") or record.get("stock_source") or "SYNTHETIC_DEMO"
        rows.append(
            {
                "product": product_name,
                "region": record.get("region"),
                "date": record.get("date"),
                "price": record.get("price"),
                "quantity": record.get("available_quantity"),
                "source_name": source,
                "data_quality": "SYNTHETIC_DEMO" if str(source).upper() in ("SYNTHETIC", "SYNTHETIC_DEMO") else "UNVERIFIED",
                "collection_method": "CSV_IMPORT",
                "unit": record.get("unit") or "kg",
            }
        )
    counts = import_market_observations(rows)
    return counts["inserted"] + counts["updated"]


def has_market_data(product_name: str = "Potato") -> bool:
    with get_conn() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT 1 FROM market_observations o JOIN products p "
                    "ON p.product_id=o.product_id WHERE p.product_name=:product LIMIT 1"
                ),
                {"product": product_name},
            ).first()
        )


def get_market_observations(
    product_name: str | None = None,
    region: str | None = None,
    start_date: Any = None,
    end_date: Any = None,
    data_quality: str | None = None,
) -> pd.DataFrame:
    clauses = []
    params: dict[str, Any] = {}
    if product_name:
        clauses.append("p.product_name = :product")
        params["product"] = product_name
    if region:
        clauses.append("o.region = :region")
        params["region"] = region
    if start_date is not None:
        clauses.append("o.observation_date >= :start_date")
        params["start_date"] = _as_date(start_date)
    if end_date is not None:
        clauses.append("o.observation_date <= :end_date")
        params["end_date"] = _as_date(end_date)
    if data_quality:
        quality = _quality_for_source("", data_quality)
        clauses.append("o.data_quality = :quality")
        params["quality"] = quality
    where_clause = "WHERE " + " AND ".join(clauses) if clauses else ""
    query = text(
        "SELECT o.observation_id, p.product_name AS product, c.category_name AS category, "
        "o.wilaya_code, o.region, o.market, o.price, o.quantity, o.incoming_supply, o.unit, "
        "o.observation_date AS date, s.source_name AS source_name, s.source_name AS data_source, "
        "o.data_quality, o.collection_method, o.notes "
        "FROM market_observations o JOIN products p ON p.product_id=o.product_id "
        "LEFT JOIN product_categories c ON c.category_id=p.category_id "
        "JOIN data_sources s ON s.source_id=o.source_id "
        f"{where_clause} ORDER BY o.observation_date, o.region, p.product_name"
    )
    with get_conn() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_latest_market_summary(product_name: str | None = None) -> pd.DataFrame:
    """Return one best-quality latest observation per product and wilaya."""
    observations = get_market_observations(product_name=product_name)
    if observations.empty:
        return observations
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    quality_order = {"OFFICIAL": 4, "VERIFIED": 3, "UNVERIFIED": 2, "SYNTHETIC_DEMO": 1}
    observations["_quality_rank"] = observations["data_quality"].map(quality_order).fillna(0)
    observations = observations.sort_values(["_quality_rank", "date"])
    latest = observations.groupby(["product", "region"], as_index=False, sort=False).tail(1)
    return latest.drop(columns="_quality_rank").reset_index(drop=True)


def deactivate_product(product_name: str) -> None:
    """Hide a product from selection without deleting its observation history."""
    with get_conn() as conn:
        result = conn.execute(
            text("UPDATE products SET is_active=FALSE WHERE lower(product_name)=lower(:name)"),
            {"name": product_name.strip()},
        )
        if result.rowcount == 0:
            raise ValueError(f"Unknown product: {product_name}")


def add_wilaya(code: str, name_fr: str, name_ar: str | None = None, name_en: str | None = None) -> None:
    """Add or update a wilaya registry entry; no geographic data is inferred."""
    normalized_code = str(code).strip().zfill(2)
    if len(normalized_code) != 2 or not normalized_code.isdigit() or not name_fr.strip():
        raise ValueError("A two-digit wilaya code and French name are required.")
    with get_conn() as conn:
        conn.execute(
            text(
                "INSERT INTO wilayas (wilaya_code, name_fr, name_ar, name_en) "
                "VALUES (:code, :name_fr, :name_ar, :name_en) "
                "ON CONFLICT (wilaya_code) DO UPDATE SET name_fr=excluded.name_fr, "
                "name_ar=excluded.name_ar, name_en=excluded.name_en"
            ),
            {
                "code": normalized_code,
                "name_fr": name_fr.strip(),
                "name_ar": name_ar,
                "name_en": name_en,
            },
        )


def save_risk_alerts(risk_rows: pd.DataFrame) -> int:
    """Persist the latest explainable risk signals without replacing prior alerts."""
    if risk_rows.empty:
        return 0
    with get_conn() as conn:
        product_cache: dict[str, int] = {}
        inserted = 0
        for row in risk_rows.to_dict(orient="records"):
            product_id = _product_id(conn, str(row["product"]), product_cache)
            indicators = {
                "price_change_pct": row.get("price_change_pct"),
                "stock_ratio": row.get("stock_ratio"),
                "unit": row.get("unit"),
                "data_quality": row.get("data_quality"),
                "reason": row.get("reason"),
            }
            exists = conn.execute(
                text(
                    "SELECT 1 FROM alerts WHERE product_id=:product_id AND region=:region "
                    "AND unit=:unit AND alert_type=:alert_type AND detected_at=:detected_at"
                ),
                {
                    "product_id": product_id,
                    "region": str(row["region"]),
                    "unit": str(row.get("unit") or ""),
                    "alert_type": str(row["status"]),
                    "detected_at": pd.to_datetime(row["date"]).to_pydatetime(),
                },
            ).first()
            if exists:
                continue
            conn.execute(
                text(
                    "INSERT INTO alerts (product_id, region, unit, alert_type, risk_level, description, "
                    "indicators_json, recommended_action, status, detected_at, data_quality) "
                    "VALUES (:product_id, :region, :unit, :alert_type, :risk_level, :description, "
                    ":indicators, :action, :status, :detected_at, :quality)"
                ),
                {
                    "product_id": product_id,
                    "region": str(row["region"]),
                    "unit": str(row.get("unit") or ""),
                    "alert_type": str(row["status"]),
                    "risk_level": str(row["risk_level"]),
                    "description": str(row["reason"]),
                    "indicators": json.dumps(indicators, ensure_ascii=False, default=str),
                    "action": str(row["recommended_action"]),
                    "status": str(row["status"]),
                    "detected_at": pd.to_datetime(row["date"]).to_pydatetime(),
                    "quality": str(row["data_quality"]),
                },
            )
            inserted += 1
        return inserted


def save_prediction_records(
    product_name: str,
    forecasts: pd.DataFrame,
    evaluation: dict[str, Any],
    horizon_days: int,
) -> int:
    if forecasts.empty:
        return 0
    with get_conn() as conn:
        product_id = _product_id(conn, product_name)
        for row in forecasts.to_dict(orient="records"):
            conn.execute(
                text(
                    "INSERT INTO predictions (product_id, region, model_name, horizon_days, "
                    "target_date, predicted_price, unit, lower_bound, upper_bound, evaluation_json) "
                    "VALUES (:product_id, :region, :model, :horizon, :target_date, :price, :unit, "
                    ":lower, :upper, :evaluation)"
                ),
                {
                    "product_id": product_id,
                    "region": str(row["region"]),
                    "model": str(row["model"]),
                    "horizon": horizon_days,
                    "target_date": _as_date(row["target_date"]),
                    "price": float(row["predicted_price"]),
                    "unit": str(row.get("unit") or "kg"),
                    "lower": float(row["lower_90"]),
                    "upper": float(row["upper_90"]),
                    "evaluation": json.dumps(evaluation, allow_nan=False),
                },
            )
    return len(forecasts)


def save_scenario_run(scenario_type: str, assumptions: dict[str, Any], results: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            text(
                "INSERT INTO scenario_runs (scenario_type, factor, assumptions_json, results_json) "
                "VALUES (:scenario, :factor, :assumptions, :results)"
            ),
            {
                "scenario": scenario_type,
                "factor": float(results.get("price_index_pct", 0)),
                "assumptions": json.dumps(assumptions, ensure_ascii=False, allow_nan=False),
                "results": json.dumps(results, ensure_ascii=False, allow_nan=False),
            },
        )


def _best_observation_query(value_column: str, alias: str) -> str:
    return (
        "WITH ranked AS (SELECT o.observation_date AS date, o.region, o.wilaya_code, o.market, "
        f"o.{value_column} AS {alias}, s.source_name AS data_source, o.data_quality, o.unit, "
        "ROW_NUMBER() OVER (PARTITION BY o.product_id, o.region, o.market, o.observation_date "
        "ORDER BY CASE o.data_quality WHEN 'OFFICIAL' THEN 4 WHEN 'VERIFIED' THEN 3 "
        "WHEN 'UNVERIFIED' THEN 2 ELSE 1 END DESC, o.observation_id DESC) AS priority "
        "FROM market_observations o JOIN products p ON p.product_id=o.product_id "
        "JOIN data_sources s ON s.source_id=o.source_id "
        "WHERE p.product_name=:product AND o."
        f"{value_column} IS NOT NULL) SELECT date, region, wilaya_code, market, {alias}, "
        "data_source, data_quality, unit, 0 AS injected_event FROM ranked "
        "WHERE priority=1 ORDER BY date, region"
    )


def get_prices(product_name: str = "Potato") -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(text(_best_observation_query("price", "price")), conn, params={"product": product_name})


def get_stock(product_name: str = "Potato") -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            text(_best_observation_query("quantity", "available_quantity")),
            conn,
            params={"product": product_name},
        )


def save_alert(
    region: str,
    alert_type: str,
    severity: str,
    description: str,
    product_name: str | None = None,
    wilaya_code: str | None = None,
    indicators: dict[str, Any] | None = None,
    recommended_action: str | None = None,
    status: str = "POTENTIAL_RISK",
    data_quality: str = "UNVERIFIED",
) -> None:
    quality = _quality_for_source("", data_quality)
    with get_conn() as conn:
        product_id = None
        if product_name:
            product_id = _product_id(conn, product_name)
        conn.execute(
            text(
                "INSERT INTO alerts (product_id, region, wilaya_code, alert_type, risk_level, "
                "description, indicators_json, recommended_action, status, data_quality) "
                "VALUES (:product_id, :region, :wilaya_code, :alert_type, :risk_level, "
                ":description, :indicators, :action, :status, :quality)"
            ),
            {
                "product_id": product_id,
                "region": region,
                "wilaya_code": wilaya_code,
                "alert_type": alert_type,
                "risk_level": severity,
                "description": description,
                "indicators": json.dumps(indicators or {}, ensure_ascii=False),
                "action": recommended_action,
                "status": status,
                "quality": quality,
            },
        )


def get_saved_alerts(limit: int = 100) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            text(
                "SELECT a.alert_id, p.product_name AS product, a.region, a.wilaya_code, a.unit, "
                "a.alert_type, a.risk_level, a.description, a.indicators_json, "
                "a.recommended_action, a.status, a.detected_at, a.data_quality "
                "FROM alerts a LEFT JOIN products p ON p.product_id=a.product_id "
                "ORDER BY a.detected_at DESC LIMIT :limit"
            ),
            conn,
            params={"limit": max(1, min(int(limit), 1000))},
        )


def save_prediction(
    product_name: str,
    model_name: str,
    horizon_days: int,
    predicted_price: float,
    target_date: Any = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    evaluation: dict[str, Any] | None = None,
    region: str = "National",
) -> None:
    if horizon_days not in (7, 30) or predicted_price < 0:
        raise ValueError("Prediction horizon must be 7 or 30 days and price must be non-negative.")
    with get_conn() as conn:
        product_id = _product_id(conn, product_name)
        conn.execute(
            text(
                "INSERT INTO predictions (product_id, region, model_name, horizon_days, target_date, "
                "predicted_price, lower_bound, upper_bound, evaluation_json) "
                "VALUES (:product_id, :region, :model_name, :horizon, :target_date, :price, "
                ":lower, :upper, :evaluation)"
            ),
            {
                "product_id": product_id,
                "region": region,
                "model_name": model_name,
                "horizon": horizon_days,
                "target_date": _as_date(target_date) if target_date is not None else None,
                "price": predicted_price,
                "lower": lower_bound,
                "upper": upper_bound,
                "evaluation": json.dumps(evaluation or {}, ensure_ascii=False),
            },
        )


def save_scenario(scenario_type: str, factor: float, assumptions: dict[str, Any], results: dict[str, Any]) -> int:
    with get_conn() as conn:
        conn.execute(
            text(
                "INSERT INTO scenario_runs (scenario_type, factor, assumptions_json, results_json) "
                "VALUES (:scenario_type, :factor, :assumptions, :results)"
            ),
            {
                "scenario_type": scenario_type,
                "factor": factor,
                "assumptions": json.dumps(assumptions, ensure_ascii=False),
                "results": json.dumps(results, ensure_ascii=False),
            },
        )
        return int(conn.execute(text("SELECT MAX(scenario_id) FROM scenario_runs")).scalar_one())


def get_last_update(table: str) -> str | None:
    allowed = {"prices": "market_observations", "stock": "market_observations", "market_observations": "market_observations"}
    table_name = allowed.get(table)
    if table_name is None:
        raise ValueError("Unsupported table name.")
    with get_conn() as conn:
        value = conn.execute(text(f"SELECT MAX(observation_date) FROM {table_name}")).scalar_one_or_none()
    return str(value) if value is not None else None
