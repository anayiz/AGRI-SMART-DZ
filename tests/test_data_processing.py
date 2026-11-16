import unittest

import pandas as pd

from src.data_processing import chrono_split


class ChronologicalSplitTests(unittest.TestCase):
    def test_training_targets_do_not_cross_into_test_period(self):
        dates = pd.date_range("2025-12-01", "2026-01-20", freq="D")
        frame = pd.DataFrame({"date": dates})
        features = pd.DataFrame({"value": range(len(dates))})
        targets = pd.Series(range(len(dates)))

        X_train, X_test, _, _, train_rows, test_rows = chrono_split(
            frame, features, targets, split_date="2026-01-01", horizon=7
        )

        self.assertEqual(len(X_train), len(train_rows))
        self.assertEqual(len(X_test), len(test_rows))
        self.assertLess(train_rows["date"].max() + pd.Timedelta(days=7), pd.Timestamp("2026-01-01"))
        self.assertGreaterEqual(test_rows["date"].min(), pd.Timestamp("2026-01-01"))


if __name__ == "__main__":
    unittest.main()
