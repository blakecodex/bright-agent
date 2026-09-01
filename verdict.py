"""
verdict.py - turn evidence into a call. deterministic, explainable, bounded.

the mock model cannot reason, so the reasoning lives here in code where it can
be read, tested and argued with. three price signals and one market dial:

  comps    - list price vs the median of comparable closed sales (the appraiser's move)   [vote]
  model    - list price vs the hedonic + mlp estimate (the statistician's move)          [vote]
  ppsf     - $/sqft vs size-matched comps: a sanity check on size, not a third vote       [check]
  market   - days on market and months of supply widen or narrow the tolerance band     [dial]

why ppsf is a check and not a vote: small homes carry a high $/sqft by nature, so
the ratio disagrees with the median for honest reasons. it earns the right to lower
our confidence, not to move the price.

output: verdict in {fairly_priced, overpriced, underpriced, insufficient_data},
a confidence in [0, 1], and reasons written as numbers a broker can repeat.
"""

VERDICTS = ("fairly_priced", "overpriced", "underpriced", "insufficient_data")
BASE_BAND = 0.05          # +/- 5% of the reference price is "fair" in a balanced market
MIN_COMPS = 3             # fewer than this and the median is an anecdote
CONFIDENCE_FLOOR = 0.55   # below this we do not issue a call; a human does


def pct(a, b):
    """signed percent difference of a over b, or None when b is missing."""
    if a is None or b in (None, 0):
        return None
    return (a - b) / b


def assess(listing, comps=None, market=None, model=None):
    comps = comps or {}
    market = market or {}
    model = model or {}
    reasons, signals, weights = [], {}, {}

    if not isinstance(listing, dict) or not listing:
        return _out("insufficient_data", 0.0, ["no property record - nothing to price"], signals)
    price = listing.get("list_price")
    if not price:
        return _out("insufficient_data", 0.0, ["no list price on the listing"], signals)

    # ---- signal 1: comps median
    n = comps.get("comp_count") or 0
    if n >= MIN_COMPS and comps.get("median_sale_price"):
        d = pct(price, comps["median_sale_price"])
        signals["comps_delta"] = d
        # more comps, more trust: saturates around 12
        weights["comps_delta"] = min(1.0, n / 12.0)
        reasons.append(f"list ${price:,.0f} vs comps median ${comps['median_sale_price']:,.0f} "
                       f"(n={n}, {comps.get('window_months', 12)}mo): {d:+.1%}")
    elif n:
        reasons.append(f"only {n} comps in window - median not trusted")
    elif "error" in comps:
        reasons.append(f"comps unavailable: {comps['error']}")

    # ---- check: price per square foot against size-matched comps (feeds confidence only)
    sqft = listing.get("sqft")
    ref_ppsf = comps.get("median_ppsf_similar") or comps.get("median_ppsf")
    k = comps.get("similar_size_count") if comps.get("median_ppsf_similar") else n
    if sqft and ref_ppsf and (k or 0) >= MIN_COMPS:
        d = pct(price / sqft, ref_ppsf)
        signals["ppsf_delta"] = d
        tag = "size-matched comps" if comps.get("median_ppsf_similar") else "comps"
        reasons.append(f"size check: ${price / sqft:,.0f}/sqft vs {tag} ${ref_ppsf:,.0f}/sqft (k={k}): {d:+.1%}")

    # ---- signal 2: the model
    if model.get("predicted_price"):
        d = pct(price, model["predicted_price"])
        signals["model_delta"] = d
        spread = (model.get("model_spread_pct") or 0) / 100.0
        w = 0.8 if model.get("zip_in_training_vocab") else 0.4
        w *= max(0.3, 1.0 - spread)          # two models that disagree get less say
        weights["model_delta"] = w
        reasons.append(f"model estimate ${model['predicted_price']:,.0f} "
                       f"(ridge/mlp spread {model.get('model_spread_pct', 0):.0f}%): {d:+.1%}")
    elif "error" in model:
        reasons.append(f"model unavailable: {model['error']}")

    if not weights:
        return _out("insufficient_data", 0.0, reasons + ["no usable price signal - route to analyst"], signals)

    # ---- combine: weighted mean of the votes, each clipped so one wild number cannot run the show
    total = sum(weights.values())
    combined = sum(max(-0.5, min(0.5, signals[k])) * weights[k] for k in weights) / total
    signals["combined_delta"] = combined

    # ---- market dial: sellers' markets forgive premiums, buyers' markets punish them
    band = BASE_BAND
    regime = market.get("regime")
    if regime == "sellers":
        band = 0.07
    elif regime == "buyers":
        band = 0.03
    if market.get("median_dom"):
        reasons.append(f"market: {market.get('region', '?')} median dom {market['median_dom']:.0f} days, "
                       f"{market.get('months_of_supply', 0):.1f} months supply, {regime or 'unknown'} market, "
                       f"dom {market.get('dom_trend', 'flat')} -> tolerance band +/-{band:.0%}")
    dom = listing.get("days_on_market")
    if dom is not None and market.get("median_dom"):
        if dom > 1.5 * market["median_dom"]:
            combined += 0.02   # sitting on the market is the market voting "overpriced"
            reasons.append(f"listing dom {dom} > 1.5x market median - nudged toward overpriced")
        elif dom < 0.5 * market["median_dom"]:
            reasons.append(f"listing dom {dom} is fresh - no dom penalty")

    # ---- verdict
    if combined > band:
        verdict = "overpriced"
    elif combined < -band:
        verdict = "underpriced"
    else:
        verdict = "fairly_priced"

    # ---- confidence: evidence volume x agreement (votes and the size check) x distance from the line
    volume = min(1.0, total / 1.8)
    deltas = [signals[k] for k in list(weights) + (["ppsf_delta"] if "ppsf_delta" in signals else [])]
    agreement = 1.0 - min(1.0, (max(deltas) - min(deltas)) / 0.30) if len(deltas) > 1 else 0.6
    margin = min(1.0, abs(abs(combined) - band) / band)  # how far from the fair/unfair boundary
    confidence = round(0.30 + 0.25 * volume + 0.30 * agreement + 0.15 * margin, 2)
    if confidence < CONFIDENCE_FLOOR:
        reasons.append(f"confidence {confidence:.2f} below floor {CONFIDENCE_FLOOR} - route to analyst")
        return _out("insufficient_data", confidence, reasons, signals, provisional=verdict)
    reasons.append(f"combined delta {combined:+.1%} vs band +/-{band:.0%} -> {verdict} (confidence {confidence:.2f})")
    return _out(verdict, confidence, reasons, signals)


def _out(verdict, confidence, reasons, signals, provisional=None):
    out = {"verdict": verdict, "confidence": confidence, "reasons": reasons,
           "signals": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in signals.items()}}
    if provisional:
        out["provisional_verdict"] = provisional
    return out


def gate(result, threshold=CONFIDENCE_FLOOR):
    """the human-in-the-loop decision. returns (route, why)."""
    if result["verdict"] == "insufficient_data" or result["confidence"] < threshold:
        return "analyst", "low confidence or missing evidence"
    return "auto", "confidence above threshold and evidence complete"


if __name__ == "__main__":
    demo = assess({"list_price": 415_000, "sqft": 1650, "days_on_market": 12},
                  comps={"comp_count": 4, "median_sale_price": 402_500, "window_months": 3})
    print(demo["verdict"], demo["confidence"]); print("\n".join(" - " + r for r in demo["reasons"]))
