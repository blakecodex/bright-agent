"""
features.py - turn a property record into a numeric vector. the hedonic idea:
a house is a bundle of attributes, and price is (roughly) additive in the logs.

one FeatureSpec is fitted on the training rows (zip vocabulary, scaler means and
stds, imputation medians) and then frozen. train and predict must build vectors
the same way or the weights mean nothing - so both call spec.transform.
"""

import math

import numpy as np

# opa grades run a+ (best) to e; we map to an ordinal where smaller is better
QUALITY_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "E+", "E", "E-"]
QUALITY_RANK = {g: i for i, g in enumerate(QUALITY_ORDER)}
DEFAULT_QUALITY = QUALITY_RANK["C"]  # the modal philadelphia rowhome

BUILDING_FAMILIES = ["ROW", "TWIN", "DET"]   # detached is the catch-all for everything else
CURRENT_YEAR = 2026


def building_family(desc):
    # 'ROW PORCH FRONT' -> 'ROW', 'TWIN CONVENTIONAL' -> 'TWIN', 'CONVENTIONAL' -> 'DET', '' -> 'DET'
    if not desc:
        return "DET"
    head = str(desc).split()[0].upper()
    return head if head in ("ROW", "TWIN") else "DET"


def _num(x, default=None):
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


class FeatureSpec:
    """fit once, transform many. serialisable to plain json so the agent can load it without numpy pickles."""

    NUMERIC = ["log_sqft", "beds", "baths", "age", "log_lot", "stories", "ext_cond", "quality", "garage",
               "fireplaces", "central_air", "air_missing", "is_mf", "beds_missing", "log_assessed", "assessed_missing"]
    # log_assessed is the city's own valuation. it is public for every parcel before a listing goes
    # live, so it is fair game as a feature - an avm that ignores the assessor is leaving money on the table.

    def __init__(self, min_zip_count=15, hidden_target=None):
        self.min_zip_count = min_zip_count
        self.zips = []          # vocabulary of one-hot zips; anything else is "other"
        self.medians = {}       # imputation values for numeric fields
        self.mean = None        # scaler
        self.std = None
        self.names = []

    def _med(self, key, default):
        # a fitted median when we have one, else a sensible philadelphia default
        return self.medians.get(key) or default

    # ---- record -> raw dict of numbers (before scaling)
    def raw(self, rec):
        sqft = _num(rec.get("sqft"), None)
        beds = _num(rec.get("beds"), None)
        beds_missing = 1.0 if not beds else 0.0           # opa uses 0 for "unknown"
        beds = beds if beds else self._med("beds", 3.0)
        baths = _num(rec.get("baths"), None) or self._med("baths", 1.0)
        yb = _num(rec.get("year_built"), None) or self._med("year_built", 1925.0)
        lot = _num(rec.get("lot_sqft"), None) or self._med("lot_sqft", 1000.0)
        air = rec.get("central_air")
        assessed = _num(rec.get("market_value") if "market_value" in rec else rec.get("assessed_value"), None)
        assessed_missing = 0.0 if assessed and assessed > 1000 else 1.0
        if assessed_missing:
            assessed = self._med("market_value", 150000.0)
        return {
            "log_sqft": math.log(sqft if sqft and sqft > 0 else self._med("sqft", 1200.0)),
            "beds": min(beds, 8.0),
            "baths": min(baths, 6.0),
            "age": min(max(CURRENT_YEAR - yb, 0.0), 200.0),
            "log_lot": math.log(max(lot, 100.0)),
            "stories": _num(rec.get("stories"), None) or self._med("stories", 2.0),
            "ext_cond": _num(rec.get("ext_cond"), None) or 4.0,
            "quality": float(QUALITY_RANK.get(str(rec.get("quality_grade") or "").strip(), DEFAULT_QUALITY)),
            "garage": _num(rec.get("garage"), 0.0) or 0.0,
            "fireplaces": _num(rec.get("fireplaces"), 0.0) or 0.0,
            "central_air": 1.0 if air == "Y" else 0.0,
            "air_missing": 1.0 if air not in ("Y", "N") else 0.0,
            "is_mf": 1.0 if str(rec.get("cat") or rec.get("category") or "SF") == "MF" else 0.0,
            "beds_missing": beds_missing,
            "log_assessed": math.log(assessed),
            "assessed_missing": assessed_missing,
            "family": building_family(rec.get("building")),
            "zip": str(rec.get("zip") or rec.get("zip_code") or ""),
        }

    def fit(self, rows):
        # imputation medians from the observed values, ignoring zeros and blanks
        def med(key):
            vals = sorted(v for v in (_num(r.get(key)) for r in rows) if v)
            return vals[len(vals) // 2] if vals else None
        self.medians = {k: med(k) for k in ("beds", "baths", "year_built", "lot_sqft", "sqft", "stories", "market_value")}
        counts = {}
        for r in rows:
            z = str(r.get("zip") or "")
            counts[z] = counts.get(z, 0) + 1
        self.zips = sorted(z for z, c in counts.items() if z and c >= self.min_zip_count)
        self.names = self.NUMERIC + [f"fam_{f}" for f in BUILDING_FAMILIES] + [f"zip_{z}" for z in self.zips] + ["zip_other"]
        X = self._matrix(rows)
        # scale only the numeric block; one-hots stay 0/1
        k = len(self.NUMERIC)
        self.mean = np.zeros(X.shape[1]); self.std = np.ones(X.shape[1])
        self.mean[:k] = X[:, :k].mean(axis=0)
        self.std[:k] = X[:, :k].std(axis=0) + 1e-9
        return self

    def _matrix(self, rows):
        out = np.zeros((len(rows), len(self.names)))
        for i, r in enumerate(rows):
            raw = self.raw(r)
            out[i, :len(self.NUMERIC)] = [raw[n] for n in self.NUMERIC]
            j = len(self.NUMERIC)
            out[i, j + BUILDING_FAMILIES.index(raw["family"])] = 1.0
            j += len(BUILDING_FAMILIES)
            if raw["zip"] in self.zips:
                out[i, j + self.zips.index(raw["zip"])] = 1.0
            else:
                out[i, j + len(self.zips)] = 1.0
        return out

    def transform(self, rows):
        X = self._matrix(rows)
        return (X - self.mean) / self.std

    # ---- json round trip
    def to_dict(self):
        return {"min_zip_count": self.min_zip_count, "zips": self.zips, "medians": self.medians,
                "mean": self.mean.tolist(), "std": self.std.tolist(), "names": self.names}

    @classmethod
    def from_dict(cls, d):
        spec = cls(d["min_zip_count"])
        spec.zips = d["zips"]; spec.medians = d["medians"]; spec.names = d["names"]
        spec.mean = np.array(d["mean"]); spec.std = np.array(d["std"])
        return spec


def target(rows):
    # log price: turns "10% over" into a constant shift, and tames the long right tail
    return np.log(np.array([float(r["sale_price"]) for r in rows]))
