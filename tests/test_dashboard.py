import unittest

import pandas as pd

from app import ALERT_COLUMNS, build_alerts


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


if __name__ == "__main__":
    unittest.main()
