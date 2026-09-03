"""

mlp_torch.py - the same one-hidden layer network, written with pytorch.

read it next to mlp_numpy.py; the pieces map one to one:
    nn.Linear(n_in, h)  + tanh + nn.Linear(h,1)     <->  W1, b1, tanh, W2, b2
    loss.backward()                                 <-> backward() by hand
    torch.optim.Adam(...).step()                    <-> the m and v moment updates

the pair exists as a cross-check: both nets on the same data should land in the 
same place, and when they do the hand-written gradients are confirmed.

torch is optional - nothing else in the repo imports this file.

"""

import numpy as np

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional extra
    HAS_TORCH = False


class TorchMLP:
    def __init__(self, n_in, n_hidden=32, seed=0):
        if not HAS_TORCH:
            raise ImportError("pip install torch to use the torch mirror")
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(n_in, n_hidden), nn.Tanh(), nn.Linear(n_hidden, 1))
        self.y_mean, self.y_std = 0.0, 1.0
        self.history = []

    def fit(self, X, y, X_val=None, y_val=None, epochs=300, lr=2e-3, batch=64, wd=1e-4, patience=60, seed=0):
        self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-9)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor((y - self.y_mean) / self.y_std, dtype=torch.float32)[:, None]
        Xv = torch.tensor(X_val, dtype=torch.float32) if X_val is not None else None
        yv = torch.tensor((y_val - self.y_mean) / self.y_std, dtype=torch.float32)[:, None] if y_val is not None else None
        # weight_decay in adam is the same l2 term we add to the loss by hand in numpy
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(seed)
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(epochs):
            self.net.train()
            for idx in torch.randperm(len(Xt), generator=gen).split(batch):
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()      # autograd does here waht mlp_numpy.backward() does by hand
                opt.step()
            self.net.eval()
            with torch.no_grad():
                train_loss = float(loss_fn(self.net(Xt), yt))
                val_loss = float(loss_fn(self.net(Xv), yv)) if Xv is not None else train_loss
            self.history.append((epoch, train_loss, val_loss))
            if val_loss < best - 1e-5:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32))[:, 0].numpy()
        return out * self.y_std + self.y_mean


def compare_with_numpy(X, y, X_val, y_val, hidden=32, seed=0):
    """fit both on identical data and report how far apart their predictions land."""
    from .mlp_numpy import MLP
    a = MLP(X.shape[1], hidden, seed=seed).fit(X, y, X_val, y_val, epochs=300, lr=2e-3, wd=1e-4, patience=60)
    b = TorchMLP(X.shape[1], hidden, seed=seed).fit(X, y, X_val, y_val, epochs=300, lr=2e-3, wd=1e-4, patience=60)
    pa, pb = a.predict(X_val), b.predict(X_val)
    return {"numpy_rmse": float(np.sqrt(np.mean((pa - y_val) ** 2))),
            "torch_rmse": float(np.sqrt(np.mean((pb - y_val) ** 2))),
            "mean_abs_diff_between_models": float(np.mean(np.abs(pa - pb)))}
