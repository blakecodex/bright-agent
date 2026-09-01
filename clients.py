"""
clients.py - things that answer client.create(messages=..., tools=...).

three of them, one interface:
  MockClient / MalformedMockClient  - the kit's scripted turns (mock_client.py, untouched)
  PlannerClient                     - a deterministic planner for real addresses. it reads
                                      the conversation like a model would and picks the next
                                      tool by rule. it cannot reason either; it proves the
                                      scaffold on real data without a key.
  AnthropicClient                   - the real-model door. wraps the sdk and reshapes its
                                      response into the dict blocks the loop expects.

the loop never knows which one it got. that is the whole point of an interface.
"""

import json
import re

from mock_client import MockResponse


class PlannerClient:
    """
    a four-step plan, advanced by what the last tool_result said:
      lookup_listing -> (comp_stats + market_context) -> (predict_price + search_notes) -> end_turn
    if lookup fails we stop early and say so; the verdict engine then routes to a human.
    """

    def __init__(self, region="Philadelphia County, PA"):
        self.region = region
        self.calls = 0

    def create(self, messages, tools=None, model="planner-1"):
        self.calls += 1
        results = _tool_results(messages)   # name -> parsed result, in order seen
        query = messages[0]["content"] if isinstance(messages[0]["content"], str) else ""

        if "lookup_listing" not in results:
            address = _address_from(query)
            return MockResponse(
                content=[{"type": "text", "text": f"i need the property record for {address}."},
                         {"type": "tool_use", "id": f"tu_{self.calls}", "name": "lookup_listing",
                          "input": {"address": address}}],
                stop_reason="tool_use")

        listing = results["lookup_listing"]
        if "error" in listing:
            return MockResponse(content=[{"type": "text", "text": f"no record found: {listing['error']}. "
                                          "i cannot price what i cannot find - routing to an analyst."}],
                                stop_reason="end_turn")

        if "comp_stats" not in results:
            beds = listing.get("beds") or 3
            return MockResponse(
                content=[{"type": "text", "text": "now the comps and the market backdrop."},
                         {"type": "tool_use", "id": f"tu_{self.calls}a", "name": "comp_stats",
                          "input": {"zip_code": listing.get("zip_code"), "beds": beds, "sqft": listing.get("sqft")}},
                         {"type": "tool_use", "id": f"tu_{self.calls}b", "name": "market_context",
                          "input": {"region": self.region}}],
                stop_reason="tool_use")

        if "predict_price" not in results:
            return MockResponse(
                content=[{"type": "text", "text": "a model estimate to triangulate, and the method note to cite."},
                         {"type": "tool_use", "id": f"tu_{self.calls}a", "name": "predict_price",
                          "input": {"property": listing}},
                         {"type": "tool_use", "id": f"tu_{self.calls}b", "name": "search_notes",
                          "input": {"query": "how the pricing verdict is made comps median band confidence", "k": 1}}],
                stop_reason="tool_use")

        comps, market, model_out = results["comp_stats"], results["market_context"], results["predict_price"]
        parts = [f"{listing.get('address')} ({listing.get('beds')} bd / {listing.get('baths')} ba, "
                 f"{listing.get('sqft')} sqft, built {listing.get('year_built')}), priced at ${listing.get('list_price', 0):,.0f}."]
        if comps.get("comp_count"):
            parts.append(f"{comps['comp_count']} comparable sales in {comps.get('zip_code')} over "
                         f"{comps.get('window_months', 12)} months, median ${comps['median_sale_price']:,.0f}.")
        if market.get("median_dom"):
            parts.append(f"{market.get('region')} is a {market.get('regime')} market "
                         f"(median dom {market['median_dom']:.0f}, {market.get('months_of_supply')} months supply).")
        if model_out.get("predicted_price"):
            parts.append(f"model estimate ${model_out['predicted_price']:,.0f}.")
        parts.append("the verdict engine below weighs these; a human signs off when confidence is low.")
        return MockResponse(content=[{"type": "text", "text": " ".join(parts)}], stop_reason="end_turn")


class AnthropicClient:
    """
    real model, same interface. requires `pip install anthropic` and ANTHROPIC_API_KEY.
    the sdk returns objects with attributes; the loop wants dicts with "type" keys, so we
    convert here and nowhere else. tool schemas pass straight through (same json shape).
    """

    SYSTEM = ("you are a pricing analyst for an mls. use the tools to gather a property record, comparable "
              "sales, market context and a model estimate before answering. be brief and numeric. never invent "
              "figures; if a tool errors, say so and stop. a separate verdict engine makes the final call.")

    def __init__(self, model="claude-sonnet-4-5", max_tokens=800):
        import anthropic  # imported lazily so the stdlib path never needs it
        self._client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def create(self, messages, tools=None, model=None):
        resp = self._client.messages.create(model=model or self.model, max_tokens=self.max_tokens,
                                            system=self.SYSTEM, messages=messages, tools=tools or [])
        content = []
        for block in resp.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        return MockResponse(content=content, stop_reason=resp.stop_reason)


# ---------------------------------------------------------------- helpers

def _tool_results(messages):
    """rebuild name -> result from the transcript: pair each tool_use id with its tool_result."""
    names, out = {}, {}
    for m in messages:
        if isinstance(m.get("content"), str):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                names[b.get("id")] = b.get("name")
            elif b.get("type") == "tool_result":
                name = names.get(b.get("tool_use_id"))
                try:
                    out[name] = json.loads(b.get("content") or "{}")
                except (TypeError, ValueError):
                    out[name] = {"error": "unparseable tool result"}
    return out


def _address_from(query):
    # "is 3358 livingston st fairly priced?" -> "3358 livingston st"; falls back to the whole query
    m = re.search(r"\b(\d{1,6}\s+[a-z0-9 .'-]+?)(?:\s+(?:listed|fairly|priced|worth|at|for|in)\b|\?|$)", query, re.I)
    return (m.group(1) if m else query).strip(" ?.,")


def make_client(kind="planner", **kw):
    if kind == "mock":
        from mock_client import MockClient
        return MockClient()
    if kind == "broken":
        from mock_client import MalformedMockClient
        return MalformedMockClient()
    if kind == "real":
        return AnthropicClient(**kw)
    return PlannerClient(**kw)
