"""
tracing.py - the run log. one json line per event, one file per run.

 - named tracing, not trace, because python already has a trace module.

 - each run records: 
        - the question 
        - every model call with its stop reason and timing
        - every tool call with input
        - output and errors
        - anything a guardrail cut
        - the critic's flags
        - and where teh run was routed


observability tools like langsmith ingest exactly this shape, so exploring later should be straightforward.

    with Tracer("traces") as t:
        t.event("model_call", turn=1, stop_reason="tool_use", latency_ms=12)

    python bright.py trace traces/<run>.jsonl # readable summary
"""

import json
import os
import time
import uuid


class Tracer:
    def __init__(self, folder="traces", run_name="run", enabled=True):
        self.enabled = enabled
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.path = os.path.join(folder, f"{run_name}-{self.run_id}.jsonl")
        self._fh = None
        self._t0 = time.time()
        self.events = []      # kept in memory too, so eval can inspect a run without re-reading

    def __enter__(self):
        if self.enabled:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self.event("run_start")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.event("run_end", ok=exc is None, error=str(exc) if exc else None,
                   total_ms=round(1000 * (time.time() - self._t0)))
        if self._fh:
            self._fh.close()
        return False   # never swallow the exception; the trace just records it

    def event(self, kind, **fields):
        rec = {"ts": round(time.time(), 3), "run_id": self.run_id, "kind": kind}
        rec.update(_shrink(fields))
        self.events.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec, default=str) + "\n")
            self._fh.flush()
        return rec


    def timed(self):
        """returns a closure that reports elapsed milliseconds - `done = t.timed(); ...; done()`"""
        start = time.time()
        return lambda: round(1000 * (time.time() - start), 1)


def _shrink(fields, limit=600):
    # tool outputs can be big (a comps list). keep the trace readable: cap long strings.
    out = {}
    for k, v in fields.items():
        s = json.dumps(v, default=str)
        out[k] = v if len(s) <= limit else {"_truncated": s[:limit] + "...", "_bytes": len(s)}
    return out


def summarize(path):
    """print a readable timeline for one run file."""
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    t0 = rows[0]["ts"] if rows else 0
    print(f"run {rows[0]['run_id']}  events {len(rows)}  file {os.path.basename(path)}")
    for r in rows:
        dt = f"+{r['ts'] - t0:6.3f}s"
        kind = r["kind"]
        if kind == "model_call":
            print(f"{dt}  model   turn={r.get('turn')} stop={r.get('stop_reason')} {r.get('latency_ms')}ms")
        elif kind == "tool_call":
            status = "error" if r.get("error") else "ok"
            print(f"{dt}  tool    {r.get('name')}({_short(r.get('input'))}) -> {status} {r.get('latency_ms')}ms")
        elif kind == "guardrail":
            print(f"{dt}  guard   {r.get('stage')}: {r.get('detail')}")
        elif kind == "gate":
            print(f"{dt}  gate    -> {r.get('route')} ({r.get('why')})")
        elif kind == "run_end":
            print(f"{dt}  end     ok={r.get('ok')} total={r.get('total_ms')}ms")
        else:
            print(f"{dt}  {kind}")


def _short(obj, n=60):
    s = json.dumps(obj, default=str) if obj is not None else ""
    return s if len(s) <= n else s[:n] + "..."


if __name__ == "__main__":
    import sys
    summarize(sys.argv[1])
