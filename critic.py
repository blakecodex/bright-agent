"""
critic.py - the second pair of eyes: planner -> worker -> critic.

 - the loop (planner) decides which tools to call, the tools (workers) fetch and
 compute, and this module reads the finished verdict against the raw evidence,

    - it asks: is this internally consistent? it does not re-derive the answer;
      it checks the answer's story.

    - this is the "self-reflection" step in the multi-agent pattern, kept deterministic
      so it cannot hallucinate a second opinion.


 - review() is the rule-based judge. 
 - llm_judge() is the slot where a real model scores the same rubric - 
   same inputs, same pass/fail contract - 
   so swapping in "llm-as-a-judge" later changes one function, not the loop.
"""

import re

RUBRIC = [
    "every dollar figure in the reasons appears in the evidence",
    "the verdict agrees with the sign of the combined delta",
    "confidence is not high when the comp count is low",
    "insufficient_data is used when no price signal exists, and only then",
]


def review(result, evidence):
    flags = []
    signals = result.get("signals", {})
    comps = evidence.get("comp_stats") or {}
    n = comps.get("comp_count") or 0
    combined = signals.get("combined_delta")

    # every dollar amount quoted in the reasons has to exist somewhere in the evidence
    quoted = _dollars(" ".join(result.get("reasons", [])))
    known = _numbers(str(evidence))
    listing = evidence.get("lookup_listing") or {}
    if listing.get("list_price"):
        known.add(round(float(listing["list_price"])))
    for amount in quoted:
        if amount >= 1000 and not any(abs(amount - k) <= max(1000, 0.01 * k) for k in known):
            flags.append(f"${amount:,.0f} in reasons not traceable to evidence")

    # the label has to match the direction of the number
    if combined is not None and result["verdict"] in ("overpriced", "underpriced"):
        if (combined > 0) != (result["verdict"] == "overpriced"):
            flags.append("verdict direction contradicts combined delta")

    # few comps should not come with high confidence
    if n and n < 5 and result.get("confidence", 0) > 0.8:
        flags.append(f"confidence {result['confidence']} too high for {n} comps")

    # and the reverse: don't refuse to call it when the signals are there
    if result["verdict"] == "insufficient_data" and combined is not None and result.get("confidence", 0) >= 0.75:
        flags.append("insufficient_data issued despite usable signals")

    return {"pass": not flags, "flags": flags, "rubric": RUBRIC}


def llm_judge(result, evidence, client=None):
    """optional model-based review of the same rubric. without a client it falls back to
    review(). with one, the model's flags are added to the rule flags, never substituted -
    the rules always run."""
    base = review(result, evidence)
    if client is None:
        return base
    prompt = ("you are auditing a pricing verdict. rubric:\n- " + "\n- ".join(RUBRIC) +
              f"\n\nverdict: {result}\n\nevidence: {evidence}\n\nanswer PASS or FAIL, then one line per problem.")
    resp = client.create(messages=[{"role": "user", "content": prompt}], tools=[])
    text = "\n".join(b.get("text", "") for b in resp.content if b.get("type") == "text")
    llm_flags = [ln.strip("- ").strip() for ln in text.splitlines()[1:] if ln.strip()] if text.upper().startswith("FAIL") else []
    return {"pass": base["pass"] and not llm_flags, "flags": base["flags"] + llm_flags, "rubric": RUBRIC,
            "judge_text": text}


def _dollars(text):
    # amounts written with a dollar sign, like "$402,500"
    return {round(float(m.replace(",", ""))) for m in re.findall(r"\$([\d,]+(?:\.\d+)?)", text)}


def _numbers(text):
    # every bare number in the evidence dump
    return {round(float(m)) for m in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", text)}