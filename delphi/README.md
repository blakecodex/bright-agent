# Delphi — the web front for bright-agent

Delphi is what a listing agent would actually open: one page, four predictive actions, every number shown with the evidence beneath it, and a person making the final call. It runs on top of the agent in this repo — no logic is duplicated into the web layer — and deploys to Render as one free web service.

## What it does

| action | question it answers | what runs |
|---|---|---|
| **Price check** | Is this asking price fair, and how sure are we? | the full agent loop (`agent.run` with the planner), then the verdict engine, critic and gate |
| **Price ladder** | At what asking price does the verdict flip? | the same evidence re-scored by `verdict.assess` at eleven prices from −15% to +15% — no new tool calls |
| **Market pulse** | What is the county doing right now? | `market_context` for one of 13 Bright-footprint counties, a 12-month table and a sparkline |
| **Ask the method** | How is any of this computed? | `retrieval.search_notes` over `notes/` — every answer is a quoted paragraph with its source |

Under the verdict: the record, comps (with the size-matched $/sqft check), the market, the model (ridge, mlp, spread), the critic's rubric, the method note it cited, and the full trace with latencies. Two buttons: **copy seller summary** (plain text for an email) and **download trace** (json).

Why "predictive actions" and not "briefs": a brief is a document about a property; an agent wants an answer to a question. Each button is a question with a numeric answer and a confidence, which is also what the JD's "predictive" means in practice.

## Run it locally

```bash
pip install -r delphi/requirements.txt        # numpy + flask (+ gunicorn, unused on windows)
python -m delphi.app                          # http://127.0.0.1:8000
```

The sqlite file is built lazily on the first request from the gzipped pages in `data/cache/`, so a fresh clone works without a fetch. The trained model is read from `ml/artifacts/model.json`.

## Deploy to Render

1. Push this repo to GitHub.
2. In Render: **New +** → **Blueprint** → connect the repo. Render reads `render.yaml` at the repo root (build `pip install -r delphi/requirements.txt`, start `gunicorn delphi.app:app`, health check `/api/health`).
3. Deploy. First build ~2 minutes; the free tier sleeps after 15 idle minutes and wakes on the next request (~30 s).
4. Optional: add `ANTHROPIC_API_KEY` in the service's environment to enable the **claude** narrator. The verdict does not depend on it; without the key the radio button is disabled and the page says so.

Manual alternative (no blueprint): New + → Web Service → runtime Python → build command `pip install -r delphi/requirements.txt` → start command `gunicorn delphi.app:app --bind 0.0.0.0:$PORT`.

## The api (what the page calls)

```
GET  /api/health                      {ok, sales_rows, market_rows, narrator, python}
POST /api/price-check  {address, price?, dom?, narrator?}
                                      {verdict, route, route_reason, critic, evidence, ladder, trace, turns, elapsed_ms, model_text, answer}
POST /api/market       {region}       {context, history[12], notes}
POST /api/ask-method   {question, k?} {hits[]: id, source, title, text, score}
GET  /api/regions                     {regions[]}
```

Inputs go through the same guardrails as the cli (`guardrails.check_question`, `check_query`): a blocked input returns `400` with the reason. Nothing is written to disk per request; the tracer runs in memory and its events are returned in the response.

## Files

```
delphi/
├── app.py                 flask app: five routes, the ladder sweep, helpers
├── templates/index.html   the page: actions, forms, guidance tables with tooltips, trust panel
├── static/delphi.css      one stylesheet, no framework; print-friendly
├── static/delphi.js       vanilla js: status line, rendering, sparkline, copy/download
├── requirements.txt       -r ../requirements.txt + flask + gunicorn
└── README.md
render.yaml                at the repo root: the blueprint render reads
```

## Limits, stated

- City deed records, not MLS: no list prices or listing days on market. The form's asking price and dom are the seam.
- 2,000 sales in the cache; the models and comps are honest about that in their confidence. `python bright.py fetch` pulls the full window.
- The planner cannot reason; with a key, a real model plans and narrates and the verdict stays deterministic.
- The regime and band rules are the broker's rules of thumb, labelled as such in the ui.
- Render's free tier has an ephemeral disk: the sqlite file is rebuilt on each cold start (about a second).
