import unittest

from data import store


class Store(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = store.ensure_db()

    def test_tables_have_rows(self):
        n = self.con.execute("SELECT count(*) FROM sales").fetchone()[0]
        m = self.con.execute("SELECT count(*) FROM market").fetchone()[0]
        self.assertGreater(n, 1000)
        self.assertGreater(m, 500)

    def test_sql_median_matches_python(self):
        # the window-function median must agree with a plain python median on the same rows
        out = store.comps("19134", 3, con=self.con)
        prices = sorted(r["sale_price"] for r in self.con.execute(
            "SELECT sale_price FROM sales WHERE zip='19134' AND beds=3 AND sale_date > date(?, '-12 months') AND sale_date <= ?",
            (out["as_of"], out["as_of"])).fetchall())
        n = len(prices)
        py = prices[n // 2] if n % 2 else (prices[n // 2 - 1] + prices[n // 2]) / 2
        self.assertEqual(out["comp_count"], n)
        self.assertAlmostEqual(out["median_sale_price"], py, delta=1)

    def test_no_comps_is_zero_not_error(self):
        out = store.comps("00000", 3, con=self.con)
        self.assertEqual(out["comp_count"], 0)
        self.assertIsNone(out["median_sale_price"])

    def test_find_property_forgives_case_and_commas(self):
        self.assertIsNotNone(store.find_property("3358 livingston st, philadelphia", con=self.con))

    def test_market_partial_month_dropped(self):
        m = store.market_context(con=self.con)
        self.assertGreater(m["homes_sold"], 100)   # a partial month would be tiny


if __name__ == "__main__":
    unittest.main()
