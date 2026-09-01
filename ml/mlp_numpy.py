"""
mlp_numpy.py - a one-hidden-layer network with the gradients written out by hand.

    forward:  z1 = X W1 + b1      a1 = tanh(z1)      yhat = a1 W2 + b2
    loss:     L  = mean((yhat - y)^2) + wd * (|W1|^2 + |W2|^2)
    backward: chain rule, one line per arrow, read bottom to top

why tanh: bounded, smooth, zero-centred; on 2k rows relu tends to leave dead
units and the extra expressiveness buys nothing. why one hidden layer: the
universal approximation result says width is enough, and depth is the thing
that needs lots of data. this is the smallest net that can bend the hedonic
line - which is all we are asking it to do.

the optimiser is adam, written out too (m and v running moments, bias-corrected).
"""

import numpy as np


class MLP:
    def __init__(self, n_in, n_hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        # glorot-ish scale: keeps tanh in its linear-ish regime at init
        self.W1 = rng.normal(0, np.sqrt(2.0 / (n_in + n_hidden)), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / (n_hidden + 1)), (n_hidden, 1))
        self.b2 = np.zeros(1)
        self.y_mean, self.y_std = 0.0, 1.0   # we standardise the target too
        self.history = []

    # ---- the two directions
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        yhat = (a1 @ self.W2 + self.b2)[:, 0]
        return yhat, a1

    def backward(self, X, a1, yhat, y, wd):
        n = X.shape[0]
        d_yhat = 2.0 * (yhat - y) / n                    # dL/dyhat
        dW2 = a1.T @ d_yhat[:, None] + 2 * wd * self.W2   # dL/dW2 = a1' * d_yhat
        db2 = d_yhat.sum(keepdims=True)
        d_a1 = d_yhat[:, None] @ self.W2.T               # push the error back through W2
        d_z1 = d_a1 * (1.0 - a1 ** 2)                    # tanh'(z) = 1 - tanh(z)^2
        dW1 = X.T @ d_z1 + 2 * wd * self.W1
        db1 = d_z1.sum(axis=0)
        return dW1, db1, dW2, db2

    def loss(self, X, y, wd=0.0):
        yhat, _ = self.forward(X)
        mse = float(((yhat - y) ** 2).mean())
        return mse + wd * (float((self.W1 ** 2).sum()) + float((self.W2 ** 2).sum()))

    # ---- training
    def fit(self, X, y, X_val=None, y_val=None, epochs=300, lr=3e-3, batch=64, wd=1e-4,
            patience=30, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-9)
        yt = (y - self.y_mean) / self.y_std
        yv = (y_val - self.y_mean) / self.y_std if y_val is not None else None

        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        best, best_state, bad, step = np.inf, None, 0, 0

        for epoch in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch):
                idx = order[start:start + batch]
                yhat, a1 = self.forward(X[idx])
                grads = self.backward(X[idx], a1, yhat, yt[idx], wd)
                step += 1
                for i, (p, g) in enumerate(zip(params, grads)):
                    m[i] = b1 * m[i] + (1 - b1) * g
                    v[i] = b2 * v[i] + (1 - b2) * g * g
                    m_hat = m[i] / (1 - b1 ** step)
                    v_hat = v[i] / (1 - b2 ** step)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)   # in-place: params alias the arrays
            train_loss = self.loss(X, yt)
            val_loss = self.loss(X_val, yv) if yv is not None else train_loss
            self.history.append((epoch, train_loss, val_loss))
            if verbose and epoch % 25 == 0:
                print(f"epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}")
            # early stopping: keep the weights from the best validation epoch
            if val_loss < best - 1e-5:
                best, bad = val_loss, 0
                best_state = [p.copy() for p in params]
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.W1, self.b1, self.W2, self.b2 = best_state
        return self

    def predict(self, X):
        yhat, _ = self.forward(X)
        return yhat * self.y_std + self.y_mean

    # ---- persistence
    def to_dict(self):
        return {"W1": self.W1.tolist(), "b1": self.b1.tolist(), "W2": self.W2.tolist(), "b2": self.b2.tolist(),
                "y_mean": self.y_mean, "y_std": self.y_std}

    @classmethod
    def from_dict(cls, d):
        m = cls(len(d["W1"]), len(d["W1"][0]))
        m.W1, m.b1 = np.array(d["W1"]), np.array(d["b1"])
        m.W2, m.b2 = np.array(d["W2"]), np.array(d["b2"])
        m.y_mean, m.y_std = d["y_mean"], d["y_std"]
        return m


def gradient_check(n=5, d=4, h=3, eps=1e-6, seed=1):
    """
    finite differences vs the analytic gradient. if this passes, the backward
    pass is right - it is the unit test every hand-written network deserves.
    returns the largest relative error over a handful of weights.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)); y = rng.normal(size=n)
    net = MLP(d, h, seed=seed)
    yhat, a1 = net.forward(X)
    dW1, db1, dW2, db2 = net.backward(X, a1, yhat, y, wd=0.0)
    worst = 0.0
    for P, G in ((net.W1, dW1), (net.b1, db1), (net.W2, dW2), (net.b2, db2)):
        for _ in range(5):
            i = tuple(rng.integers(0, s) for s in P.shape)
            old = P[i]
            P[i] = old + eps; lp = net.loss(X, y)
            P[i] = old - eps; lm = net.loss(X, y)
            P[i] = old
            numeric = (lp - lm) / (2 * eps)
            worst = max(worst, abs(numeric - G[i]) / max(1e-8, abs(numeric) + abs(G[i])))
    return worst


if __name__ == "__main__":
    print("max relative gradient error:", gradient_check())
