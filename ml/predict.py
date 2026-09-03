"""

predict.py - the inference side: load model.json once and answers.
"what should this property sell for?" with two estimates and their spread.

kept separate from train.py so serving never imports the training loop.

"""

import json
import math
import os

from .features import FeatureSpec
from .linear import Ridge
from .mlp_numpy import MLP

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "artifacts", "model.json")

_cache = {}


def load(path=MODEL_PATH):
    if path not in _cache:
        if not os.path.exists(path):
            raise FileNotFoundError("no trained model at ml/artifacts/model.json - run `python -m ml.train`")
        with open(path) as fh:
            blob = json.load(fh)
        _cache[path] = (FeatureSpec.from_dict(blob["spec"]), Ridge.from_dict(blob["ridge"]),
                        MLP.from_dict(blob["mlp"]), blob.get("version", "?"), blob.get("report", {}))
    return _cache[path]


def predict(property_record, path=MODEL_PATH):
    """
    property_record: the dict lookup_listing returns (zip_code, beds, baths, sqft, ...).
    returns prices in dollars plus the spread between the two models.
    """
    try:
        spec, ridge, mlp, version, report = load(path)
    except FileNotFoundError as e:
        return {"error": str(e)}
    if not isinstance(property_record, dict) or not property_record.get("sqft"):
        return {"error": "predict_price needs a property record with at least sqft and zip_code"}

    X = spec.transform([property_record])
    p_ridge = math.exp(float(ridge.predict(X)[0]))
    p_mlp = math.exp(float(mlp.predict(X)[0]))
    blend = math.sqrt(p_ridge * p_mlp)                    # geometric mean: averaging in log space
    spread = abs(p_ridge - p_mlp) / blend
    zip_known = str(property_record.get("zip_code") or property_record.get("zip")) in spec.zips
    return {
        "predicted_price": round(blend, -3),
        "ridge_price": round(p_ridge, -3),
        "mlp_price": round(p_mlp, -3),
        "model_spread_pct": round(100 * spread, 1),
        "zip_in_training_vocab": zip_known,
        "model_version": version,
        "holdout_mape": {"ridge": report.get("ridge", {}).get("mape"), "mlp": report.get("mlp", {}).get("mape")},
        "note": "hedonic ridge + one-hidden-layer mlp on philadelphia sales; log-price target",
    }
