"""
verdict.py - turns the evidence into a verdict; plain arithmetic, no model.

two votes - one check - one dial:
- comps     - asking price vs the median of comparable closed sales (vote)
- model     - asking price vs the two price models' estiamte (vote)
- ppsf      - price per sqft vs similar-sized comps; affects confidence only (check)
- market    - days on market and months of supply set the tolerance band (dial)

ppsf only checks because small homes naturally cost more per sqft (not an assumption - verifiable in the data);
a real mismatch should lower confidence, not drag the price.

returns a verdict (fairly_priced, overpriced, underpriced, insufficent_data),
a confidence between 0 and 1, and the reasons for the broker.
"""

VERDICTS = ("fairly_priced", "overpriced", "underpriced", "insufficient_data")
BASE_BAND = 0.05          # +/- 5% of the reference price is "fair" in a balanced market
MIN_COMPS = 3             # fewer than this and the median is an anecdote
CONFIDENCE_FLOOR = 0.55   # below this we do not issue a call; a human does


def pct(a, b):
    """signed percent difference of a vs b. None if b is missing"""
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

    # comps vote
    n = comps.get("comp_count") or 0
    if n >= MIN_COMPS and comps.get("median_sale_price"):
        d = pct(price, comps["median_sale_price"])
        signals["comps_delta"] = d
        weights["comps_delta"] = min(1.0, n / 12.0)  # weight grows with comp count, caps at 12
        reasons.append(f"list ${price:,.0f} vs comps median ${comps['median_sale_price']:,.0f} "
                       f"(n={n}, {comps.get('window_months', 12)}mo): {d:+.1%}")
    elif n:
        reasons.append(f"only {n} comps in window - median not trusted")
    elif "error" in comps:
        reasons.append(f"comps unavailable: {comps['error']}")

    # size check: $/sqft against similar-sized comps. affects confidence only.
    sqft = listing.get("sqft")
    ref_ppsf = comps.get("median_ppsf_similar") or comps.get("median_ppsf")
    k = comps.get("similar_size_count") if comps.get("median_ppsf_similar") else n
    if sqft and ref_ppsf and (k or 0) >= MIN_COMPS:
        d = pct(price / sqft, ref_ppsf)
        signals["ppsf_delta"] = d
        tag = "size-matched comps" if comps.get("median_ppsf_similar") else "comps"
        reasons.append(f"size check: ${price / sqft:,.0f}/sqft vs {tag} ${ref_ppsf:,.0f}/sqft (k={k}): {d:+.1%}")

    # model vote: discounted when ridge and mlp disagree or the zip is unfamiliar
    if model.get("predicted_price"):
        d = pct(price, model["predicted_price"])
        signals["model_delta"] = d
        spread = (model.get("model_spread_pct") or 0) / 100.0
        w = 0.8 if model.get("zip_in_training_vocab") else 0.4
        w *= max(0.3, 1.0 - spread)
        weights["model_delta"] = w
        reasons.append(f"model estimate ${model['predicted_price']:,.0f} "
                       f"(ridge/mlp spread {model.get('model_spread_pct', 0):.0f}%): {d:+.1%}")
    elif "error" in model:
        reasons.append(f"model unavailable: {model['error']}")

    if not weights:
        return _out("insufficient_data", 0.0, reasons + ["no usable price signal - route to analyst"], signals)

    # weighted average of the votes, each clipped to +/-50%
    total = sum(weights.values())
    combined = sum(max(-0.5, min(0.5, signals[k])) * weights[k] for k in weights) / total
    signals["combined_delta"] = combined

    # market sets the tolerance band
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
            combined += 0.02  # stale listing, nudge toward overpriced
            reasons.append(f"listing dom {dom} > 1.5x market median - nudged toward overpriced")
        elif dom < 0.5 * market["median_dom"]:
            reasons.append(f"listing dom {dom} is fresh - no dom penalty")

    if combined > band:
        verdict = "overpriced"
    elif combined < -band:
        verdict = "underpriced"
    else:
        verdict = "fairly_priced"

    # confidence: how much evidence, how well it agrees, how far from the band
    volume = min(1.0, total / 1.8)
    deltas = [signals[k] for k in list(weights) + (["ppsf_delta"] if "ppsf_delta" in signals else [])]
    agreement = 1.0 - min(1.0, (max(deltas) - min(deltas)) / 0.30) if len(deltas) > 1 else 0.6
    margin = min(1.0, abs(abs(combined) - band) / band)
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
    """route the result: auto when confident and complete, analyst otherwise."""
    if result["verdict"] == "insufficient_data" or result["confidence"] < threshold:
        return "analyst", "low confidence or missing evidence"
    return "auto", "confidence above threshold and evidence complete"


if __name__ == "__main__":
    demo = assess({"list_price": 415_000, "sqft": 1650, "days_on_market": 12},
                  comps={"comp_count": 4, "median_sale_price": 402_500, "window_months": 3})
    print(demo["verdict"], demo["confidence"])
    print("\n".join(" - " + r for r in demo["reasons"]))

