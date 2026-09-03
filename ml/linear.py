"""
linear.py - ridge regression, solved in closed form.

least squar with a penalty on large weights.

    (X'X + lam*I) w = X'y

the solution comes from the normal equations, and the code solves the linear system
rather than inverting the matrix - same answer, better numerics.

the intercept is a column of ones and is not penalized.

"""

import numpy as np


class Ridge:
    def __init__(self, lam=1.0):
        self.lam = lam
        self.w = None      # weights, intercept last

    def fit(self, X, y):
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])
        reg = self.lam * np.eye(d + 1)
        reg[d, d] = 0.0                       # do not shrink the intercept
        # normal equations: (X'X + lam I) w = X'y
        self.w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        return self

    def predict(self, X):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return Xb @ self.w

    def coef_table(self, names, top=12):
        # largest coefficients first: which attributes move price, in log units
        pairs = sorted(zip(names, self.w[:-1]), key=lambda p: -abs(p[1]))
        return pairs[:top]

    def to_dict(self):
        return {"lam": self.lam, "w": self.w.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["lam"])
        m.w = np.array(d["w"])
        return m


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0
