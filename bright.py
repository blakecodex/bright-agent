"""

bright.py - the command line for everything beyond the two scripted runs.

    python bright.py ask "Is 3358 Livingston St fairly priced?"
    python bright.py ask "Is 720 Shirley St fairly priced?" -- price 499000 --dom
    python bright.py ask "Is 123 Oak St listed at a fair price?" --client mock
    python bright.py eval                # scenario harness, prints the scorecard
    python bright.py train               # refit ridge + mlp, write ml/artifacts/model.json
    python bright.py evaluate            # k-fold comparison incl. sklearn / torch cross-checks
    python bright.py fetch [--since ..]  # refresh data/cache from the live apis
    python bright.py db                  # rebuild the sqlite file from cache
    python bright.py trace <file>        # print one run log
    python bright.py notes "months of supply"
    
each subcommand is a thin wrapper around one module.
"""

import argparse
import json
import sys

import guardrails


def cmd_ask(args):
    import agent
    import clients
    from tracing import Tracer
    try:
        guardrails.check_question(args.query)
        _, price, dom = guardrails.check_query(None, args.price, args.dom)
    except guardrails.GuardrailError as e:
        print(f"blocked by input guardrail: {e}")
        return 2
    client = clients.make_client(args.client)
    result = agent.run(client, args.query, tracer=Tracer("traces", args.client, enabled=not args.no_trace),
                       list_price=price, days_on_market=dom)
    print(result["answer"])
    if args.json:
        print(json.dumps({k: result[k] for k in ("verdict", "route", "critic", "turns")}, indent=2, default=str))
    if result["trace_path"]:
        print(f"\ntrace: {result['trace_path']}")
    return 0


def cmd_eval(args):
    import eval as harness
    return 0 if harness.main(verbose=True) else 1


def cmd_train(args):
    from ml import train
    train.main()
    return 0


def cmd_evaluate(args):
    from ml import evaluate_models
    evaluate_models.main(folds=args.folds)
    return 0


def cmd_fetch(args):
    from data import fetch_philly, fetch_redfin, store
    rc = fetch_philly.main(["--since", args.since] if args.since else [])
    if not args.skip_redfin:
        rc |= fetch_redfin.main([])
    store.build()
    return rc


def cmd_db(args):
    from data import store
    store.build()
    return 0


def cmd_trace(args):
    from tracing import summarize
    summarize(args.path)
    return 0


def cmd_notes(args):
    import retrieval
    for h in retrieval.search_notes(args.query, k=args.k)["hits"]:
        print(f"{h['score']:.3f}  {h['id']}\n    {h['text']}\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bright", description="bright-agent command line")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask", help="run the agent on a question")
    p.add_argument("query")
    p.add_argument("--price", type=float, help="hypothetical list price (overrides last sale)")
    p.add_argument("--dom", type=int, help="listing days on market")
    p.add_argument("--client", default="planner", choices=["planner", "mock", "broken", "real"])
    p.add_argument("--json", action="store_true", help="also print the structured result")
    p.add_argument("--no-trace", action="store_true")
    p.set_defaults(fn=cmd_ask)

    sub.add_parser("eval", help="run the scenario harness").set_defaults(fn=cmd_eval)
    sub.add_parser("train", help="fit the models").set_defaults(fn=cmd_train)
    p = sub.add_parser("evaluate", help="k-fold model comparison")
    p.add_argument("--folds", type=int, default=5)
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("fetch", help="refresh the data cache from the public apis")
    p.add_argument("--since")
    p.add_argument("--skip-redfin", action="store_true")
    p.set_defaults(fn=cmd_fetch)
    sub.add_parser("db", help="rebuild sqlite from cache").set_defaults(fn=cmd_db)
    p = sub.add_parser("trace", help="print a trace file")
    p.add_argument("path")
    p.set_defaults(fn=cmd_trace)
    p = sub.add_parser("notes", help="search the method notes")
    p.add_argument("query") 
    p.add_argument("-k", type=int, default=3)
    p.set_defaults(fn=cmd_notes)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
