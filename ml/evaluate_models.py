"""
evaluate_models.py - k-fold cross-validation, the honest scoreboard.

one held-out split can flatter or embarrass a model by luck. k folds average the
luck out. every candidate sees the same folds, built from the same feature spec,
so the comparison is about the model and nothing else:

  zip median      - "what did 3-beds in this zip go for?" the broker's back-of-envelope
  assessor        - the city's own market value, straight from the record
  ridge (ours)    - closed-form hedonic regression, ml/linear.py
  mlp (ours)      - one hidden layer, hand-written backprop, ml/mlp_numpy.py
  sklearn ridge   - the library on the same features: if it disagrees with ours, we have a bug
  sklearn gbrt    - gradient-boosted trees: the strong tabular baseline to know where the ceiling is
  torch mlp       - same architecture via autograd, when torch is installed

    python -m ml.evaluate_models --folds 5
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np

from data import store
from .features import FeatureSpec, target
from .linear import Ridge, r2
from .mlp_numpy import MLP

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "artifacts", "eval_results.json")


def fold_of(row, k):
    return int(hashlib.md5(str(row["parcel"]).encode()).hexdigest(), 16) % k


def metrics(y_log, yhat_log):
    y, yhat = np.exp(y_log), np.exp(yhat_log)
    ape = np.abs(yhat - y) / y
    return {"r2_log": r2(y_log, yhat_log), "mape": float(ape.mean()), "median_ape": float(np.median(ape)),
            "within_10pct": float((ape <= 0.10).mean())}


def zip_median_predictor(train_rows, test_rows):
    # group by (zip, beds) on the training fold; fall back to zip, then to the global median
    from collections import defaultdict
    by_zip_beds, by_zip, everything = defaultdict(list), defaultdict(list), []
    for r in train_rows:
        p = float(r["sale_price"])
        by_zip_beds[(r["zip"], r["beds"])].append(p); by_zip[r["zip"]].append(p); everything.append(p)
    med = lambda xs: float(np.median(xs))
    out = []
    for r in test_rows:
        pool = by_zip_beds.get((r["zip"], r["beds"])) or by_zip.get(r["zip"]) or everything
        out.append(np.log(med(pool)))
    return np.array(out)


def main(folds=5, verbose=True):
    rows = store.training_rows()
    names = ["zip_median", "assessor", "ridge", "mlp"]
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import Ridge as SkRidge
        names += ["sk_ridge", "sk_gbrt"]
        has_sk = True
    except ImportError:
        has_sk = False
    try:
        from .mlp_torch import TorchMLP, HAS_TORCH
        if HAS_TORCH:
            names.append("torch_mlp")
    except ImportError:
        HAS_TORCH = False

    preds = {n: [] for n in names}
    truth = []
    t0 = time.time()
    for k in range(folds):
        test = [r for r in rows if fold_of(r, folds) == k]
        train = [r for r in rows if fold_of(r, folds) != k]
        # a slice of train steers early stopping so the test fold stays untouched
        val = [r for r in train if fold_of(r, 97) < 10]
        fit = [r for r in train if fold_of(r, 97) >= 10]
        spec = FeatureSpec().fit(fit)
        Xf, yf = spec.transform(fit), target(fit)
        Xv, yv = spec.transform(val), target(val)
        Xt, yt = spec.transform(test), target(test)
        truth.append(yt)

        preds["zip_median"].append(zip_median_predictor(train, test))
        mv = np.array([float(r["market_value"] or 0) for r in test])
        preds["assessor"].append(np.where(mv > 0, np.log(np.maximum(mv, 1)), np.median(yf)))
        preds["ridge"].append(Ridge(lam=3.0).fit(Xf, yf).predict(Xt))
        preds["mlp"].append(MLP(Xf.shape[1], 32, seed=k).fit(Xf, yf, Xv, yv, epochs=600, lr=2e-3, wd=1e-4, patience=60).predict(Xt))
        if has_sk:
            preds["sk_ridge"].append(SkRidge(alpha=3.0).fit(Xf, yf).predict(Xt))
            gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=k)
            preds["sk_gbrt"].append(gb.fit(Xf, yf).predict(Xt))
        if "torch_mlp" in names:
            preds["torch_mlp"].append(TorchMLP(Xf.shape[1], 32, seed=k).fit(Xf, yf, Xv, yv, epochs=600, lr=2e-3, wd=1e-4, patience=60).predict(Xt))
        if verbose:
            print(f"fold {k + 1}/{folds}: train {len(fit)} val {len(val)} test {len(test)}")

    y = np.concatenate(truth)
    table = {n: metrics(y, np.concatenate(p)) for n, p in preds.items()}
    result = {"ran_at": time.strftime("%Y-%m-%d %H:%M:%S"), "folds": folds, "n_rows": len(rows),
              "seconds": round(time.time() - t0, 1), "models": table}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    if verbose:
        print(f"\n{folds}-fold cv on {len(rows)} sales ({result['seconds']}s)")
        print(f"{'model':12s} {'r2(log)':>8s} {'mape':>7s} {'med|err|':>9s} {'<=10%':>7s}")
        for n, m in table.items():
            print(f"{n:12s} {m['r2_log']:8.3f} {m['mape']:7.3f} {m['median_ape']:9.3f} {m['within_10pct']:7.1%}")
        print(f"saved {os.path.relpath(OUT_PATH)}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    main(folds=ap.parse_args().folds)
