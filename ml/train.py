"""
train.py - fit the statistical layer (ridge) and the hidden layer (mlp) on the
sales in the store, then freeze both into ml/artifacts/model.json.

    python -m ml.train

the split is by parcel hash, not random: the same house sold twice should not
sit on both sides of the line. the two models see identical features so their
disagreement is informative - when they diverge the verdict lowers its confidence.
"""

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
ARTIFACTS = os.path.join(HERE, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "model.json")


def split_by_parcel(rows, holdout=0.2, val=0.1):
    # deterministic: hash the parcel id, take the low bits. same split every run, every machine.
    # three buckets: test is never touched during training; val steers early stopping.
    train, val_rows, test = [], [], []
    for r in rows:
        h = int(hashlib.md5(str(r["parcel"]).encode()).hexdigest(), 16) % 1000
        if h < holdout * 1000:
            test.append(r)
        elif h < (holdout + val) * 1000:
            val_rows.append(r)
        else:
            train.append(r)
    return train, val_rows, test


def mape(y_price, yhat_price):
    return float(np.mean(np.abs(yhat_price - y_price) / y_price))


def fit_all(rows, hidden=32, verbose=True, seed=0):
    train_rows, val_rows, test_rows = split_by_parcel(rows)
    spec = FeatureSpec().fit(train_rows)
    Xtr, ytr = spec.transform(train_rows), target(train_rows)
    Xva, yva = spec.transform(val_rows), target(val_rows)
    Xte, yte = spec.transform(test_rows), target(test_rows)

    t0 = time.time()
    ridge = Ridge(lam=3.0).fit(Xtr, ytr)
    mlp = MLP(Xtr.shape[1], hidden, seed=seed).fit(Xtr, ytr, Xva, yva, epochs=600, lr=2e-3, wd=1e-4,
                                                   patience=60, verbose=False)
    elapsed = time.time() - t0

    report = {}
    for name, model in (("ridge", ridge), ("mlp", mlp)):
        yhat = model.predict(Xte)
        report[name] = {
            "r2_log": round(r2(yte, yhat), 3),
            "mape": round(mape(np.exp(yte), np.exp(yhat)), 3),
            "median_abs_pct_err": round(float(np.median(np.abs(np.exp(yhat) - np.exp(yte)) / np.exp(yte))), 3),
        }
    # the city's own assessment as a yardstick: how good is the assessor as a price model?
    mv = np.array([r["market_value"] or 0 for r in test_rows], dtype=float)
    ok = mv > 0
    report["assessor_baseline"] = {
        "r2_log": round(r2(yte[ok], np.log(mv[ok])), 3),
        "mape": round(mape(np.exp(yte[ok]), mv[ok]), 3),
        "median_abs_pct_err": round(float(np.median(np.abs(mv[ok] - np.exp(yte[ok])) / np.exp(yte[ok]))), 3),
    }
    report["n_train"], report["n_val"], report["n_test"] = len(train_rows), len(val_rows), len(test_rows)
    report["n_features"] = Xtr.shape[1]
    report["train_seconds"] = round(elapsed, 1)
    report["mlp_epochs_run"] = len(mlp.history)

    if verbose:
        print(f"train {len(train_rows)}  val {len(val_rows)}  test {len(test_rows)}  features {Xtr.shape[1]}  "
              f"({elapsed:.1f}s, mlp stopped after {len(mlp.history)} epochs)")
        for k in ("assessor_baseline", "ridge", "mlp"):
            print(f"  {k:18s} r2(log)={report[k]['r2_log']:.3f}  mape={report[k]['mape']:.3f}  "
                  f"median|err|={report[k]['median_abs_pct_err']:.3f}")
        print("  top ridge coefficients (log-price units):")
        for name, w in ridge.coef_table(spec.names, top=10):
            print(f"    {name:16s} {w:+.3f}")
    return spec, ridge, mlp, report


def save(spec, ridge, mlp, report, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {"version": time.strftime("%Y-%m-%d"), "spec": spec.to_dict(), "ridge": ridge.to_dict(),
            "mlp": mlp.to_dict(), "report": report}
    with open(path, "w") as fh:
        json.dump(blob, fh)
    return path


def main():
    rows = store.training_rows()
    spec, ridge, mlp, report = fit_all(rows)
    out = save(spec, ridge, mlp, report)
    print(f"saved {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
