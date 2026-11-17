import unittest

import numpy as np
import pandas as pd

from src import market_intelligence as intelligence


class MarketIntelligenceTests(unittest.TestCase):
    def test_forecast_uses_chronological_holdout_and_compares_baselines(self):
        dates = pd.date_range("2025-01-01", periods=240, freq="D")
        observations = pd.DataFrame(
            {
                "date": dates,
                "region": "Alger",
                "price": 40 + np.arange(len(dates)) * 0.08 + np.sin(np.arange(len(dates)) / 9),
                "quantity": 500 - np.arange(len(dates)) * 0.2,
            }
        )

        result = intelligence.forecast_product(observations, horizon=7)

        self.assertEqual(result["status"], "ok")
        self.assertLess(pd.Timestamp(result["training_target_end"]), pd.Timestamp(result["split_date"]))
        self.assertGreaterEqual(
            result["predictions"]["date"].min(), pd.Timestamp(result["split_date"])
        )
        self.assertIn("Persistence baseline", result["metrics"])
        self.assertIn("Random Forest", result["metrics"])
        self.assertIn("Linear Regression", result["metrics"])
        self.assertIn("Gradient Boosting", result["metrics"])
        self.assertEqual(result["forecast"]["target_date"].iloc[0], dates[-1] + pd.Timedelta(days=7))
        self.assertTrue({"MAE", "RMSE", "R2"}.issubset(result["metrics"]["Persistence baseline"]))

    def test_forecast_reports_insufficient_history_without_predictions(self):
        observations = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=35, freq="D"),
                "region": "Alger",
                "price": range(35),
                "quantity": [100] * 35,
            }
        )

        result = intelligence.forecast_product(observations, horizon=30)

        self.assertEqual(result["status"], "insufficient_data")
        self.assertTrue(result["forecast"].empty)
        self.assertEqual(result["metrics"], {})

    def test_forecasts_keep_different_units_in_separate_time_series(self):
        dates = pd.date_range("2025-01-01", periods=240, freq="D")
        observations = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": dates,
                        "region": "Alger",
                        "unit": unit,
                        "price": scale * (40 + np.arange(len(dates)) * 0.08),
                        "quantity": 500 - np.arange(len(dates)) * 0.2,
                    }
                )
                for unit, scale in (("kg", 1.0), ("g", 0.001))
            ],
            ignore_index=True,
        )

        result = intelligence.forecast_product(observations, horizon=7)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["forecast"]["unit"]), {"g", "kg"})
        self.assertEqual(set(result["predictions"]["unit"]), {"g", "kg"})

    def test_price_anomaly_is_not_reported_as_confirmed_shortage(self):
        dates = pd.date_range("2025-01-01", periods=31, freq="D")
        observations = pd.DataFrame(
            {
                "date": dates,
                "product": "Rice",
                "region": "Alger",
                "price": [50.0] * 30 + [65.0],
                "quantity": [None] * 31,
                "data_quality": "UNVERIFIED",
            }
        )

        risks = intelligence.detect_shortage_risks(observations)

        self.assertEqual(len(risks), 1)
        self.assertEqual(risks.iloc[0]["status"], "ANOMALY")
        self.assertNotEqual(risks.iloc[0]["status"], "CONFIRMED_SHORTAGE")
        self.assertIn("above the preceding local median", risks.iloc[0]["reason"])

    def test_risk_detection_does_not_compare_prices_across_units(self):
        dates = pd.date_range("2025-01-01", periods=31, freq="D")
        observations = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": dates,
                        "product": "Rice",
                        "region": "Alger",
                        "unit": unit,
                        "price": prices,
                        "quantity": None,
                    }
                )
                for unit, prices in (
                    ("kg", [50.0] * 30 + [65.0]),
                    ("g", [0.05] * 31),
                )
            ],
            ignore_index=True,
        )

        risks = intelligence.detect_shortage_risks(observations)

        self.assertEqual(risks["unit"].tolist(), ["kg"])
        self.assertEqual(risks.iloc[0]["status"], "ANOMALY")

    def test_combined_price_and_stock_signals_raise_potential_risk(self):
        dates = pd.date_range("2025-01-01", periods=31, freq="D")
        observations = pd.DataFrame(
            {
                "date": dates,
                "product": "Rice",
                "region": "Alger",
                "price": [50.0] * 30 + [65.0],
                "quantity": [200.0] * 30 + [50.0],
                "data_quality": "VERIFIED",
            }
        )

        risks = intelligence.detect_shortage_risks(observations)

        self.assertEqual(risks.iloc[0]["risk_level"], "RED")
        self.assertEqual(risks.iloc[0]["status"], "POTENTIAL_RISK")
        self.assertIn("stock", risks.iloc[0]["reason"].lower())

    def test_simulation_is_transparent_and_validates_inputs(self):
        baseline = intelligence.simulate_scenario(100, 500)
        elevated = intelligence.simulate_scenario(100, 500, production_change_pct=-20, demand_change_pct=30)
        impact = intelligence.simulate_scenario(
            100,
            500,
            production_change_pct=-20,
            demand_change_pct=30,
            distribution_delay_days=5,
            stock_change_pct=-60,
            transport_cost_change_pct=15,
        )

        self.assertEqual(baseline["scenario_price"], 100)
        self.assertEqual(baseline["hypothetical_risk"], "NORMAL")
        self.assertEqual(elevated["hypothetical_risk"], "OBSERVE")
        self.assertEqual(impact["hypothetical_risk"], "CRITICAL")
        self.assertIn("Hypothetical", impact["label"])
        self.assertGreater(impact["scenario_price"], 100)
        with self.assertRaises(ValueError):
            intelligence.simulate_scenario(100, 500, production_change_pct=-101)
        with self.assertRaises(ValueError):
            intelligence.simulate_scenario(100, 500, distribution_delay_days=-1)


if __name__ == "__main__":
    unittest.main()
