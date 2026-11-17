from io import BytesIO, StringIO
import unittest

import pandas as pd
from openpyxl import load_workbook

from src import reporting


class ReportingTests(unittest.TestCase):
    def test_period_filter_is_inclusive_and_rejects_reversed_dates(self):
        data = pd.DataFrame(
            {
                "date": pd.date_range("2025-02-01", periods=3),
                "price": [1, 2, 3],
            }
        )

        filtered = reporting.filter_period(data, "2025-02-02", "2025-02-03")

        self.assertEqual(filtered["price"].tolist(), [2, 3])
        with self.assertRaises(ValueError):
            reporting.filter_period(data, "2025-02-03", "2025-02-01")

    def test_spreadsheet_exports_escape_formula_like_text(self):
        data = pd.DataFrame({"product": ["=1+1", "safe"], "price": [10.0, 0.0]})

        csv = reporting.to_csv_bytes(data).decode("utf-8-sig")
        csv_frame = pd.read_csv(StringIO(csv))
        workbook = load_workbook(BytesIO(reporting.to_excel_bytes([("Market", data)])))

        self.assertEqual(csv_frame.loc[0, "product"], "'=1+1")
        self.assertEqual(workbook["Market"]["A2"].value, "'=1+1")
        self.assertEqual(workbook["Market"]["B2"].value, 10)

    def test_comparison_series_keeps_units_separate(self):
        data = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01"],
                "product": ["Rice", "Rice"],
                "unit": ["kg", "g"],
                "price": [50.0, 0.05],
            }
        )

        comparison = reporting.comparison_series(data, "product")

        self.assertEqual(set(comparison["unit"]), {"g", "kg"})
        self.assertEqual(sorted(comparison["price"]), [0.05, 50.0])


if __name__ == "__main__":
    unittest.main()
