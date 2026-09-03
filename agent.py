"""
agent.py - the agent loop: it orchestrates, no data access, no math, and no model weight.

the signature run.py depends on:
run_agent(client, user_query) -> str
keep the message history; call the model with it; when it requests tool calls,
execute them, append the model's turn and the tool results, and call again;
when it answers, return the text; hard stop after ten turns.  

around that signature: tracing logs every model call and tool call, guardrails scan tool output
before the model reads it, results are kept by name as evidence, and after the loop a verdict
is computed, reviewed by a critic, and routed - auto-cleared or human-in-the-loop to an analyst.
"""

import json
import critic
import guardrails
import verdict as verdict_engine
from tools import TOOL_SCHEMAS, execute_tool
from tracing import Tracer

MAX_TURNS = 10


def run(client, user_query, tracer=None, list_price=None, days_on_market=None, trace_dir="traces"):
    """run one question end to end. returns a dict: answer text, verdict, route, evidence, transcript."""
    tracer = tracer or Tracer(trace_dir, enabled=False)
    messages = [{"role": "user", "content": user_query}]
    evidence = {}            # tool name -> last successful result
    final_text = None
    turns = 0

    with tracer as t:
        t.event("query", text=user_query, list_price=list_price, days_on_market=days_on_market)
        for turn in range(MAX_TURNS):
            turns = turn + 1
            done = t.timed()
            response = client.create(messages=messages, tools=TOOL_SCHEMAS)   # send the full history, get one reply
            t.event("model_call", turn=turns, stop_reason=response.stop_reason, latency_ms=done())

            if response.stop_reason == "end_turn":
                final_text = _text_of(response.content)
                break

            # the model asked for tools. run each one, collect the results, send them back in one user turn.
            results = []
            for block in response.content:
                if block.get("type") != "tool_use":
                    continue
                name, tool_input = block.get("name"), block.get("input") or {}
                done = t.timed()
                result = execute_tool(name, tool_input)
                if name == "lookup_listing" and isinstance(result, dict) and "error" not in result:
                    if list_price is not None:
                        result["list_price"] = list_price     # caller supplied a hypothetical asking price
                    if days_on_market is not None:
                        result["days_on_market"] = days_on_market
                result, findings = guardrails.scan_tool_output(result)
                if findings:
                    t.event("guardrail", stage="tool_output", detail=f"redacted {findings} in {name}")
                error = result.get("error") if isinstance(result, dict) else None
                t.event("tool_call", name=name, input=tool_input, error=error, latency_ms=done(),
                        output=result if error else _summary(result))
                if not error:
                    evidence[name] = result
                results.append({"type": "tool_result", "tool_use_id": block.get("id"),
                                "content": json.dumps(result, default=str)})

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": results})
        else:
            final_text = "iteration cap hit - no final answer"
            t.event("guardrail", stage="loop", detail=f"{MAX_TURNS}-turn cap reached")

        # after the loop: the verdict is computed from the evidence. the model's text is narration, not the decision.
        result = verdict_engine.assess(evidence.get("lookup_listing", {}), comps=evidence.get("comp_stats"),
                                       market=evidence.get("market_context"), model=evidence.get("predict_price"))
        problems = guardrails.check_verdict(result)
        if problems:
            t.event("guardrail", stage="output", detail=problems)
            result = {"verdict": "insufficient_data", "confidence": 0.0, "reasons": problems, "signals": {}}
        review = critic.review(result, evidence)
        t.event("critic", passed=review["pass"], flags=review["flags"])
        route, why = verdict_engine.gate(result)
        if not review["pass"] and route == "auto":
            route, why = "analyst", "critic flagged: " + "; ".join(review["flags"])
        t.event("gate", route=route, why=why, verdict=result["verdict"], confidence=result["confidence"])

    answer = compose(final_text, result, route, why, evidence.get("search_notes"))
    return {"answer": answer, "model_text": final_text, "verdict": result, "route": route, "route_reason": why,
            "critic": review, "evidence": evidence, "turns": turns, "transcript": messages,
            "trace_path": tracer.path if tracer.enabled else None}


def run_agent(client, user_query):
    """the signature run.py depends on: question in, answer text out."""
    return run(client, user_query)["answer"]


# --- helpers below:

def _text_of(content):
    # join the text blocks of a reply; ignore tool_use blocks
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text").strip()


def _summary(result):
    # comps lists are long; log the count instead of the rows
    if isinstance(result, dict) and "comps" in result:
        r = dict(result); r["comps"] = f"[{len(result['comps'])} rows]"; return r
    return result


def compose(model_text, result, route, why, notes=None):
    """format the final answer: model text, then the verdict with its reasons, the route, one cited note."""
    lines = []
    if model_text:
        lines += [model_text, ""]
    lines.append(f"verdict engine: {result['verdict'].replace('_', ' ')} (confidence {result['confidence']:.2f})")
    lines += [f"  - {r}" for r in result["reasons"]]
    lines.append(f"route: {route} - {why}")
    if notes and notes.get("hits"):
        h = notes["hits"][0]
        lines.append(f"method note ({h['id']}): {h['text'][:140].rstrip()}...")
    return "\n".join(lines)