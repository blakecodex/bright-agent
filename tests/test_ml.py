import unittest

import numpy as np

from ml.features import FeatureSpec, building_family
from ml.linear import Ridge
from ml.mlp_numpy import MLP, gradient_check


class Features(unittest.TestCase):
    def test_building_family(self):
        self.assertEqual(building_family("ROW PORCH FRONT"), "ROW")
        self.assertEqual(building_family("TWIN CONVENTIONAL"), "TWIN")
        self.assertEqual(building_family(""), "DET")

    def test_transform_is_deterministic_and_finite(self):
        rows = [{"parcel": str(i), "zip": "19134", "beds": 3, "baths": 1, "sqft": 900 + i, "lot_sqft": 1000,
                 "year_built": 1920, "sale_price": 150000 + 1000 * i, "market_value": 140000} for i in range(30)]
        spec = FeatureSpec(min_zip_count=5).fit(rows)
        X1, X2 = spec.transform(rows[:3]), spec.transform(rows[:3])
        self.assertTrue(np.allclose(X1, X2))
        self.assertTrue(np.isfinite(X1).all())


class Linear(unittest.TestCase):
    def test_ridge_recovers_a_line(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + 3.0
        m = Ridge(lam=1e-6).fit(X, y)
        self.assertTrue(np.allclose(m.w, [1.0, -2.0, 0.5, 3.0], atol=1e-3))


class Network(unittest.TestCase):
    def test_gradient_check(self):
        self.assertLess(gradient_check(), 1e-5)

    def test_learns_a_nonlinear_function(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-2, 2, size=(600, 2))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
        net = MLP(2, 24, seed=0).fit(X, y, epochs=300, lr=1e-2, batch=32, wd=0.0, patience=100)
        err = np.mean((net.predict(X) - y) ** 2)
        self.assertLess(err, 0.05)


if __name__ == "__main__":
    unittest.main()
