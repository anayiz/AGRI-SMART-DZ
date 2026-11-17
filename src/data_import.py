"""Validated CSV/XLSX ingestion for explicitly sourced market observations."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd

from . import database as db


ALIASES = {
    "product": ("product", "product_name", "commodity", "produit"),
    "region": ("region", "wilaya", "wilaya_name", "province"),
    "wilaya_code": ("wilaya_code", "code_wilaya", "code"),
    "market": ("market", "market_name", "marche"),
    "date": ("date", "observation_date", "observed_at"),
    "price": ("price", "price_dzd", "prix"),
    "quantity": ("quantity", "stock", "available_quantity", "stock_quantity"),
    "incoming_supply": ("incoming_supply", "incoming_quantity", "supply"),
    "unit": ("unit", "measurement_unit", "unite"),
    "source_name": ("source_name", "data_source", "source"),
    "data_quality": ("data_quality", "quality", "verification_status"),
    "collection_method": ("collection_method", "method"),
    "notes": ("notes", "comment", "remarks"),
}
REQUIRED = ("product", "region", "date")


def read_uploaded_file(filename: str, content: bytes) -> pd.DataFrame:
    """Parse CSV or Excel from memory; reject unsupported files clearly."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix == "csv":
        return pd.read_csv(BytesIO(content))
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(BytesIO(content))
    raise ValueError("Upload a .csv, .xlsx, or .xls file.")


def normalize_frame(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str], int]:
    """Normalize columns, validate values and find duplicate natural keys."""
    if frame.empty:
        return [], ["The uploaded file contains no data rows."], 0
    columns = {str(column).strip().lower(): column for column in frame.columns}
    rename: dict[Any, str] = {}
    for canonical, aliases in ALIASES.items():
        match = next((columns[alias] for alias in aliases if alias in columns), None)
        if match is not None:
            rename[match] = canonical
    normalized = frame.rename(columns=rename).copy()
    missing = [field for field in REQUIRED if field not in normalized]
    if missing:
        return [], [f"Missing required column(s): {', '.join(missing)}."], 0

    issues: list[str] = []
    valid: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    duplicates = 0
    for row_number, record in enumerate(normalized.to_dict(orient="records"), start=2):
        row = {key: value for key, value in record.items() if key in ALIASES}
        row = {key: (None if pd.isna(value) else value) for key, value in row.items()}
        row["product"] = str(row.get("product") or "").strip()
        row["region"] = str(row.get("region") or "").strip()
        if not row["product"] or not row["region"]:
            issues.append(f"Row {row_number}: product and wilaya/region are required.")
            continue
        try:
            row["date"] = pd.to_datetime(row.get("date"), errors="raise").date().isoformat()
            for measure in ("price", "quantity", "incoming_supply"):
                value = row.get(measure)
                if value is not None:
                    number = float(value)
                    if not pd.notna(number) or number < 0:
                        raise ValueError(f"{measure} must be a non-negative number")
                    row[measure] = number
            if not any(row.get(measure) is not None for measure in ("price", "quantity", "incoming_supply")):
                raise ValueError("price, quantity, or incoming_supply is required")
            row["source_name"] = str(row.get("source_name") or "UNVERIFIED IMPORT").strip()
            row["unit"] = str(row.get("unit") or "kg").strip()
            row["collection_method"] = str(row.get("collection_method") or "CSV_IMPORT").upper()
            if row.get("data_quality") is not None:
                quality = str(row["data_quality"]).strip().upper()
                if quality not in db.QUALITY_LEVELS:
                    raise ValueError(f"data_quality must be one of {', '.join(db.QUALITY_LEVELS)}")
                row["data_quality"] = quality
            elif row["source_name"].upper() in {"SYNTHETIC", "SYNTHETIC_DEMO", "DEMO"}:
                row["data_quality"] = "SYNTHETIC_DEMO"
            else:
                row["data_quality"] = "UNVERIFIED"
            key = (
                row["product"].casefold(),
                row["region"].casefold(),
                str(row.get("market") or "").strip().casefold(),
                row["date"],
                row["source_name"].casefold(),
            )
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            valid.append(row)
        except (TypeError, ValueError, OverflowError) as error:
            issues.append(f"Row {row_number}: {error}.")
    return valid, issues, duplicates


def import_frame(frame: pd.DataFrame) -> dict[str, Any]:
    valid, issues, duplicate_count = normalize_frame(frame)
    result = db.import_market_observations(valid)
    return {
        **result,
        "duplicates_in_file": duplicate_count,
        "invalid_rows": len(issues),
        "issues": issues[:50],
        "valid_rows": len(valid),
    }
