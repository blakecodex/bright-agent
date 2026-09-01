import unittest

import verdict


LISTING = {"list_price": 415_000, "sqft": 1650, "days_on_market": 12}
COMPS = {"comp_count": 4, "median_sale_price": 402_500, "window_months": 3}


class Verdict(unittest.TestCase):
    def test_kit_numbers_are_fair(self):
        out = verdict.assess(LISTING, comps=COMPS)
        self.assertEqual(out["verdict"], "fairly_priced")
        self.assertAlmostEqual(out["signals"]["comps_delta"], 0.031, places=3)

    def test_overpriced(self):
        out = verdict.assess(dict(LISTING, list_price=480_000), comps=dict(COMPS, comp_count=12))
        self.assertEqual(out["verdict"], "overpriced")

    def test_underpriced(self):
        out = verdict.assess(dict(LISTING, list_price=340_000), comps=dict(COMPS, comp_count=12))
        self.assertEqual(out["verdict"], "underpriced")

    def test_too_few_comps_is_insufficient(self):
        out = verdict.assess(LISTING, comps={"comp_count": 2, "median_sale_price": 400_000})
        self.assertEqual(out["verdict"], "insufficient_data")

    def test_no_listing(self):
        self.assertEqual(verdict.assess({})["verdict"], "insufficient_data")

    def test_sellers_market_widens_band(self):
        listing = dict(LISTING, list_price=427_000)   # +6.1% over the median
        balanced = verdict.assess(listing, comps=dict(COMPS, comp_count=12), market={"regime": "balanced", "median_dom": 30})
        sellers = verdict.assess(listing, comps=dict(COMPS, comp_count=12), market={"regime": "sellers", "median_dom": 30})
        self.assertEqual(balanced["verdict"], "overpriced")
        self.assertEqual(sellers["verdict"], "fairly_priced")

    def test_confidence_in_range(self):
        out = verdict.assess(LISTING, comps=COMPS)
        self.assertTrue(0.0 <= out["confidence"] <= 1.0)

    def test_gate(self):
        self.assertEqual(verdict.gate({"verdict": "insufficient_data", "confidence": 0.9})[0], "analyst")
        self.assertEqual(verdict.gate({"verdict": "overpriced", "confidence": 0.3})[0], "analyst")
        self.assertEqual(verdict.gate({"verdict": "overpriced", "confidence": 0.8})[0], "auto")


if __name__ == "__main__":
    unittest.main()
