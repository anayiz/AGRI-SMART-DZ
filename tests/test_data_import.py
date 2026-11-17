import unittest

import pandas as pd

from src.data_import import normalize_frame


class DataImportTests(unittest.TestCase):
    def test_normalizes_aliases_accepts_zero_and_counts_file_duplicates(self):
        frame = pd.DataFrame(
            [
                {"commodity": "Rice", "wilaya": "Alger", "observed_at": "2025-01-01", "prix": 0, "stock": 0},
                {"commodity": "Rice", "wilaya": "Alger", "observed_at": "2025-01-01", "prix": 0, "stock": 0},
            ]
        )

        valid, issues, duplicates = normalize_frame(frame)

        self.assertEqual(issues, [])
        self.assertEqual(duplicates, 1)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["price"], 0)
        self.assertEqual(valid[0]["quantity"], 0)
        self.assertEqual(valid[0]["data_quality"], "UNVERIFIED")

    def test_rejects_negative_values_without_losing_valid_rows(self):
        frame = pd.DataFrame(
            [
                {"product": "Milk", "region": "Oran", "date": "2025-01-01", "quantity": -1},
                {"product": "Milk", "region": "Oran", "date": "2025-01-02", "quantity": 2},
            ]
        )

        valid, issues, duplicates = normalize_frame(frame)

        self.assertEqual(len(valid), 1)
        self.assertEqual(duplicates, 0)
        self.assertEqual(len(issues), 1)
        self.assertIn("non-negative", issues[0])


if __name__ == "__main__":
    unittest.main()
