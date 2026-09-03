"""
clients.py - the three things that can play the model; all answer create(messages, tools).

    MockClient / MalformedMockClient - scripted turns from mock_client.py
    PlannerClient   - rule-based; picks the next tool from the conversation so real
                      addresses run end-to-end without an api key.
    AnthropicClient - wraps the real api and reshapes its reply into the same blocks the loop expects.


the loop cannot tell them apart; that's intentional.
"""

import json
import re

from mock_client import MockResponse


class PlannerClient:
    """
    walks a fixed four-step plan, deciding the next step from what came back so far:
    lookup -> comps + market -> prediction + notes -> summary. if the lookup fails
    it stops there and says so.
    """

    def __init__(self, region="Philadelphia County, PA"):
        self.region = region
        self.calls = 0

    def create(self, messages, tools=None, model="planner-1"):
        self.calls += 1
        results = _tool_results(messages)  # what each tool returned so far
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
                content=[{"type": "text", "text": "a model estimate, and the method note to cite."},
                         {"type": "tool_use", "id": f"tu_{self.calls}a", "name": "predict_price",
                          "input": {"property": listing}},
                         {"type": "tool_use", "id": f"tu_{self.calls}b", "name": "search_notes",
                          "input": {"query": "how the pricing verdict is made comps median band confidence", "k": 1}}],
                stop_reason="tool_use")

        # everything is in - write the summary
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
    """the real api behind the same create() signature. needs the anthropic package and ANTHROPIC_API_KEY.
    the sdk returns objects; the loop wants plain dicts, so the reply gets reshaped here."""

    SYSTEM = ("you are a pricing analyst for an mls. use the tools to gather a property record, comparable "
              "sales, market context and a model estimate before answering. be brief and numeric. never invent "
              "figures; if a tool errors, say so and stop. a separate verdict engine makes the final call.")

    def __init__(self, model="claude-sonnet-4-5", max_tokens=800):
        import anthropic  # lazy import, only this path needs it
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


def _tool_results(messages):
    # pair every tool_use id with its tool_result and parse the json
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
    # pull "3358 livingston st" out of "is 3358 livingston st fairly priced?"
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