import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src import database as db


class DatabaseImportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        db.DB_PATH = str(Path(self.temp_dir.name) / "test.db")
        db.init_db()
        db.add_product("Potato")
        self.prices_csv = Path(self.temp_dir.name) / "prices.csv"
        self.stock_csv = Path(self.temp_dir.name) / "stock.csv"
        self.prices_csv.write_text(
            "date,region,price,data_source,injected_event\n"
            "2026-07-24,Alger,55,SYNTHETIC_DEMO,0\n",
            encoding="utf-8",
        )
        self.stock_csv.write_text(
            "date,region,available_quantity,data_source,injected_event\n"
            "2026-07-24,Alger,400,SYNTHETIC_DEMO,0\n",
            encoding="utf-8",
        )

    def import_csvs(self):
        return db.load_csv_into_db(
            "Potato",
            prices_csv=str(self.prices_csv),
            stock_csv=str(self.stock_csv),
        )

    def test_import_is_idempotent_and_preserves_manual_observation(self):
        self.import_csvs()
        self.import_csvs()
        self.assertEqual(len(db.get_prices()), 1)
        self.assertEqual(len(db.get_stock()), 1)

        db.add_manual_observation("Potato", "Alger", "2026-07-24", 70, 250)
        self.import_csvs()

        prices = db.get_prices()
        stock = db.get_stock()
        self.assertEqual(len(prices), 1)
        self.assertEqual(len(stock), 1)
        self.assertEqual(prices.iloc[0]["price"], 70)
        self.assertEqual(prices.iloc[0]["data_source"], "MANUAL")
        self.assertEqual(stock.iloc[0]["available_quantity"], 250)
        self.assertEqual(stock.iloc[0]["data_source"], "MANUAL")

    def test_manual_observation_rejects_invalid_values_without_partial_write(self):
        with self.assertRaises(ValueError):
            db.add_manual_observation("Potato", "Alger", "2026-07-24", 70, -1)
        self.assertTrue(db.get_prices().empty)
        self.assertTrue(db.get_stock().empty)

    def test_concurrent_initialization_preserves_catalog_and_is_idempotent(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(db.init_db) for _ in range(8)]
            for future in futures:
                future.result(timeout=45)

        db.init_db()
        products = db.list_products()
        self.assertEqual(len(products), 15)
        self.assertIn("Potato", set(products["product_name"]))

        with db.get_conn() as conn:
            journal_mode = conn.execute(
                db.text("PRAGMA journal_mode")
            ).scalar_one()
            busy_timeout = conn.execute(
                db.text("PRAGMA busy_timeout")
            ).scalar_one()
        self.assertEqual(journal_mode.lower(), "wal")
        self.assertGreaterEqual(busy_timeout, 30000)


if __name__ == "__main__":
    unittest.main()
