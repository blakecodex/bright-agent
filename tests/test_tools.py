"""tools are proven before the agent. stdlib unittest, no fixtures beyond the cache."""
import unittest

import tools


class ToolGate(unittest.TestCase):
    def test_unknown_tool_is_data_not_exception(self):
        out = tools.execute_tool("flood_risk", {})
        self.assertIn("error", out)
        self.assertIn("unknown tool", out["error"])

    def test_bad_input_is_data_not_exception(self):
        out = tools.execute_tool("comp_stats", {"zip_code": "21043"})   # beds missing
        self.assertIn("error", out)

    def test_non_dict_input(self):
        self.assertIn("error", tools.execute_tool("comp_stats", "21043"))


class Median(unittest.TestCase):
    def test_odd(self):
        self.assertEqual(tools.median([3, 1, 2]), 2)

    def test_even(self):
        self.assertEqual(tools.median([4, 1, 3, 2]), 2.5)

    def test_empty(self):
        self.assertIsNone(tools.median([]))


class Fixtures(unittest.TestCase):
    def test_kit_comps(self):
        out = tools.comp_stats("21043", 3)
        self.assertEqual(out["median_sale_price"], 402500.0)
        self.assertEqual(out["comp_count"], 4)

    def test_kit_listing(self):
        self.assertEqual(tools.lookup_listing("123 Oak St")["list_price"], 415000)


class RealData(unittest.TestCase):
    def test_lookup_real_address(self):
        rec = tools.lookup_listing("3358 Livingston St")
        self.assertEqual(rec["zip_code"], "19134")
        self.assertEqual(rec["beds"], 3)
        self.assertTrue(rec["sqft"] > 0)

    def test_comps_have_median_and_ppsf(self):
        out = tools.comp_stats("19134", 3, sqft=884)
        self.assertGreaterEqual(out["comp_count"], 3)
        self.assertTrue(out["median_sale_price"] > 0)
        self.assertTrue(out["median_ppsf"] > 0)
        self.assertIn("median_ppsf_similar", out)

    def test_market_context_shape(self):
        m = tools.market_context()
        for key in ("median_dom", "months_of_supply", "regime", "dom_trend"):
            self.assertIn(key, m)
        self.assertIn(m["regime"], ("buyers", "sellers", "balanced"))

    def test_predict_price_returns_dollars(self):
        rec = tools.lookup_listing("3358 Livingston St")
        out = tools.predict_price(rec)
        self.assertTrue(50_000 < out["predicted_price"] < 2_000_000)
        self.assertIn("model_spread_pct", out)


if __name__ == "__main__":
    unittest.main()
