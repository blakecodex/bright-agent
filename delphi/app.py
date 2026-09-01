"""
delphi/app.py - a small web front for bright-agent, built for the person who has to
explain a price to a seller. one flask file, four endpoints, no database of its own.

delphi answers four questions a listing agent actually asks:
  price check   - is this asking price fair, and how sure are we?           (the agent loop)
  price ladder  - at what asking price does the verdict flip?               (the verdict engine, swept)
  market pulse  - what is the county doing right now?                       (the market table)
  ask the method- how is any of this computed?                              (retrieval over notes/)

everything runs keyless and deterministic. if ANTHROPIC_API_KEY is set the narrator
can be switched to a real model; the verdict never is. a person makes the final call.

    python -m delphi.app                 # local, http://127.0.0.1:8000
    gunicorn delphi.app:app              # render / any wsgi host
"""

import os
import sys
import time

from flask import Flask, jsonify, render_template, request

# the app lives one folder below the repo root; put the root on the path so the agent
# modules import exactly as they do from the cli. no code is duplicated into delphi/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agent                                  # noqa: e402  (imports after the path fix, on purpose)
import clients                                # noqa: e402
import guardrails                             # noqa: e402
import retrieval                              # noqa: e402
import tools                                  # noqa: e402
import verdict as verdict_engine              # noqa: e402
from data import store                        # noqa: e402
from tracing import Tracer                    # noqa: e402

app = Flask(__name__, template_folder="templates", static_folder="static")

DEFAULT_REGION = "Philadelphia County, PA"
LADDER_STEPS = [-0.15, -0.10, -0.07, -0.05, -0.03, 0.0, 0.03, 0.05, 0.07, 0.10, 0.15]


# ------------------------------------------------------------------ helpers

def _bad(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


def _narrator_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client(kind):
    # "rules" is the deterministic planner; "claude" is the real model when a key exists.
    if kind == "claude" and _narrator_available():
        try:
            return clients.make_client("real"), "claude"
        except Exception:      # sdk missing or key rejected: fall back rather than fail the request
            pass
    return clients.PlannerClient(region=DEFAULT_REGION), "rules"


def _trim_evidence(ev):
    # the browser does not need 8 full comp rows in two places; keep what the cards render
    out = {}
    for name, val in ev.items():
        if isinstance(val, dict):
            v = dict(val)
            if "comps" in v:
                v["comps"] = [{k: c.get(k) for k in ("address", "sale_date", "sale_price", "sqft", "ppsf")}
                              for c in v["comps"][:8]]
            out[name] = v
        else:
            out[name] = val
    return out


def _ladder(listing, comps, market, model, base_price):
    # sweep the asking price and re-run the verdict engine only - no tools, no model calls.
    # this is the "what if we asked X" question answered eleven times.
    rows = []
    for step in LADDER_STEPS:
        price = round(base_price * (1 + step), -3)
        probe = dict(listing, list_price=price)
        r = verdict_engine.assess(probe, comps=comps, market=market, model=model)
        rows.append({"step_pct": round(100 * step), "price": price, "verdict": r["verdict"],
                     "confidence": r["confidence"], "combined_delta": r["signals"].get("combined_delta"),
                     "provisional": r.get("provisional_verdict")})
    return rows


def _regions():
    con = store.ensure_db()
    rows = con.execute("SELECT DISTINCT region FROM market ORDER BY region").fetchall()
    return [r[0] for r in rows]


def _history(region, months=12):
    con = store.ensure_db()
    rows = con.execute(
        """
        SELECT period_begin, median_sale_price, median_dom, months_of_supply, inventory, homes_sold,
               avg_sale_to_list, sold_above_list
        FROM market
        WHERE region = ? AND property_type = 'All Residential'
        ORDER BY period_begin DESC LIMIT ?
        """, (region, months)).fetchall()
    return [dict(r) for r in rows][::-1]     # oldest first, so the chart reads left to right


# ------------------------------------------------------------------ routes

@app.get("/")
def index():
    return render_template("index.html", regions=_regions(), default_region=DEFAULT_REGION,
                           narrator_available=_narrator_available())


@app.get("/api/health")
def health():
    con = store.ensure_db()
    n_sales = con.execute("SELECT count(*) FROM sales").fetchone()[0]
    n_market = con.execute("SELECT count(*) FROM market").fetchone()[0]
    return jsonify({"ok": True, "sales_rows": n_sales, "market_rows": n_market,
                    "narrator": "claude available" if _narrator_available() else "rules only",
                    "python": sys.version.split()[0]})


@app.post("/api/price-check")
def price_check():
    body = request.get_json(silent=True) or {}
    address = body.get("address", "")
    try:
        question = guardrails.check_question(f"Is {address} fairly priced?")
        _, price, dom = guardrails.check_query(address, body.get("price") or None, body.get("dom") or None)
    except guardrails.GuardrailError as e:
        return _bad(f"blocked by input guardrail: {e}")

    client, narrator = _client(body.get("narrator", "rules"))
    tracer = Tracer(enabled=False)                       # events kept in memory, nothing written on the host
    t0 = time.time()
    result = agent.run(client, question, tracer=tracer, list_price=price, days_on_market=dom)
    ev = result["evidence"]

    # the ladder needs a record with a price and at least one price signal to sweep against
    ladder = []
    listing = ev.get("lookup_listing")
    if listing and listing.get("list_price") and (ev.get("comp_stats") or ev.get("predict_price")):
        ladder = _ladder(listing, ev.get("comp_stats"), ev.get("market_context"), ev.get("predict_price"),
                         float(listing["list_price"]))

    return jsonify({
        "ok": True,
        "narrator": narrator,
        "question": question,
        "answer": result["answer"],
        "model_text": result["model_text"],
        "verdict": result["verdict"],
        "route": result["route"],
        "route_reason": result["route_reason"],
        "critic": result["critic"],
        "evidence": _trim_evidence(ev),
        "ladder": ladder,
        "turns": result["turns"],
        "trace": tracer.events,
        "elapsed_ms": round(1000 * (time.time() - t0)),
    })


@app.post("/api/market")
def market():
    body = request.get_json(silent=True) or {}
    region = body.get("region") or DEFAULT_REGION
    if region not in _regions():
        return _bad("unknown region")
    ctx = tools.market_context(region)
    if "error" in ctx:
        return _bad(ctx["error"], 404)
    notes = retrieval.search_notes("months of supply days on market regime", k=2)
    return jsonify({"ok": True, "context": ctx, "history": _history(region), "notes": notes.get("hits", [])})


@app.post("/api/ask-method")
def ask_method():
    body = request.get_json(silent=True) or {}
    try:
        q = guardrails.check_question(body.get("question", ""))
    except guardrails.GuardrailError as e:
        return _bad(f"blocked by input guardrail: {e}")
    hits = retrieval.search_notes(q, k=int(body.get("k") or 4))
    return jsonify({"ok": True, "question": q, "hits": hits.get("hits", []), "note": hits.get("note")})


@app.get("/api/regions")
def regions():
    return jsonify({"ok": True, "regions": _regions()})


if __name__ == "__main__":
    # local dev only; render runs gunicorn (see render.yaml)
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
