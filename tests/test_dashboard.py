from io import BytesIO
import unittest
from unittest.mock import patch

import pandas as pd

from app import ALERT_COLUMNS, build_alerts, render_reports


class DashboardAlertTests(unittest.TestCase):
    def test_no_alerts_still_returns_expected_columns(self):
        clean_df = pd.DataFrame(
            {
                "date": pd.date_range("2026-10-01", periods=35),
                "region": ["Alger"] * 35,
                "price": [50.0] * 35,
                "stock": [100.0] * 35,
            }
        )

        alerts = build_alerts(clean_df)

        self.assertTrue(alerts.empty)
        self.assertEqual(list(alerts.columns), ALERT_COLUMNS)

    def test_report_exports_preserve_source_labels(self):
        clean_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-10-31")],
                "region": ["Alger"],
                "price": [55.0],
                "stock": [400.0],
                "data_source": ["SYNTHETIC_DEMO"],
            }
        )
        alerts = pd.DataFrame(
            [
                {
                    "region": "Alger",
                    "date": "2026-10-31",
                    "alert_level": "YELLOW",
                    "reason": "Synthetic test alert",
                    "suggested_action": "Verify the signal",
                    "price_change_pct": 6.0,
                    "stock_ratio": 0.8,
                }
            ],
            columns=ALERT_COLUMNS,
        )
        downloads = []

        with patch("app.st.subheader"), patch("app.st.caption"), patch("app.st.warning"), patch(
            "app.st.download_button",
            side_effect=lambda label, **kwargs: downloads.append((label, kwargs)),
        ):
            render_reports(clean_df, alerts)

        self.assertEqual(len(downloads), 2)
        market_export = pd.read_csv(BytesIO(downloads[0][1]["data"]))
        alert_export = pd.read_csv(BytesIO(downloads[1][1]["data"]))
        self.assertEqual(market_export.loc[0, "data_source"], "SYNTHETIC_DEMO")
        self.assertEqual(alert_export.loc[0, "alert_level"], "YELLOW")


if __name__ == "__main__":
    unittest.main()
