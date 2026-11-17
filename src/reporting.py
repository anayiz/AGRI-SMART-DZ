"""Reproducible, source-labelled market report exports."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _spreadsheet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()

    def escape(value):
        if isinstance(value, str) and value.lstrip(" \t\r\n")[:1] in ("=", "+", "-", "@"):
            return "'" + value
        return value

    for column in safe.columns:
        if safe[column].dtype == object or pd.api.types.is_string_dtype(safe[column].dtype):
            safe[column] = safe[column].map(escape)
    return safe


def filter_period(data: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date:
        raise ValueError("Report start date must be on or before its end date.")
    return result[result["date"].between(start_date, end_date)].reset_index(drop=True)


def comparison_series(data: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate prices without combining measurements with different units."""
    required = {"date", "price", "unit", dimension}
    if not required.issubset(data.columns):
        raise ValueError(f"Comparison data must include: {', '.join(sorted(required))}.")
    frame = data.dropna(subset=["price"]).copy()
    return (
        frame.groupby(["date", dimension, "unit"], as_index=False, dropna=False)["price"]
        .median()
    )


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return _spreadsheet_safe(frame).to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(sheets: Iterable[tuple[str, pd.DataFrame]]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets:
            _spreadsheet_safe(frame).to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()


def to_pdf_bytes(
    title: str,
    observations: pd.DataFrame,
    alerts: pd.DataFrame,
    start: object,
    end: object,
) -> bytes:
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=28,
        title=title,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(
            f"Period: {pd.Timestamp(start):%Y-%m-%d} to {pd.Timestamp(end):%Y-%m-%d}. "
            "Values retain their recorded source and quality status.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]
    if observations.empty:
        story.append(Paragraph("No observations are available for this period.", styles["BodyText"]))
    else:
        columns = [
            column
            for column in ("date", "product", "region", "market", "price", "quantity", "unit", "source_name", "data_quality")
            if column in observations.columns
        ]
        table_data = [columns]
        for row in observations[columns].head(60).itertuples(index=False, name=None):
            table_data.append(["" if pd.isna(value) else str(value)[:80] for value in row])
        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164B35")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DCEFE3")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7F5")]),
                ]
            )
        )
        story.extend([Paragraph("Market observations (first 60 rows)", styles["Heading2"]), table])
    if not alerts.empty:
        story.extend([Spacer(1, 12), Paragraph("Risk alerts", styles["Heading2"])])
        alert_columns = [
            column
            for column in (
                "product",
                "region",
                "unit",
                "date",
                "risk_level",
                "status",
                "reason",
                "recommended_action",
                "data_quality",
            )
            if column in alerts.columns
        ]
        alert_data = [alert_columns]
        for row in alerts[alert_columns].head(60).itertuples(index=False, name=None):
            alert_data.append(["" if pd.isna(value) else str(value)[:100] for value in row])
        alert_table = Table(alert_data, repeatRows=1)
        alert_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164B35")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DCEFE3")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7F5")]),
                ]
            )
        )
        story.append(alert_table)
    document.build(story)
    return output.getvalue()
