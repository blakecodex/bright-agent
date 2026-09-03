"""
mock_client.py — deterministic fake LLM. Cannot reason. Scripted turns only.

Mirrors the Anthropic response shape loosely:
  - response.content    -> list of blocks: {"type": "text"|"tool_use", ...}
  - response.stop_reason -> "tool_use" | "end_turn"

the loop must:
  1. call client.create(messages=history)
  2. if stop_reason == "tool_use": execute each tool_use block,
     append assistant turn + tool_result turn to history, repeat
  3. if stop_reason == "end_turn": print final text, stop
"""


class MockResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class MockClient:
    """Scenario: 'Is 123 Oak St listed at a fair price?'
    Scripted: tool_use(lookup_listing) -> tool_use(comp_stats) -> final text.
    """

    def __init__(self):
        self._turn = 0
        self._script = [
            MockResponse(
                content=[
                    {"type": "text", "text": "I need the listing details first."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "lookup_listing",
                        "input": {"address": "123 Oak St"},
                    },
                ],
                stop_reason="tool_use",
            ),
            MockResponse(
                content=[
                    {"type": "text", "text": "Now compare against recent comps."},
                    {
                        "type": "tool_use",
                        "id": "tu_2",
                        "name": "comp_stats",
                        "input": {"zip_code": "21043", "beds": 3},
                    },
                ],
                stop_reason="tool_use",
            ),
            MockResponse(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "123 Oak St is listed at $415,000. Comparable 3-bed sales "
                            "in 21043 over the last 90 days have a median of $402,500 "
                            "(n=4). The listing sits ~3% above median — within normal "
                            "range. Verdict: fairly priced; flag for human review, "
                            "not repricing."
                        ),
                    }
                ],
                stop_reason="end_turn",
            ),
        ]

    def create(self, messages, tools=None, model="mock-1"):
        # Guard: after a tool_use turn, the last message MUST carry tool_result blocks.
        if self._turn > 0:
            last = messages[-1]
            ok = (
                last.get("role") == "user"
                and isinstance(last.get("content"), list)
                and all(b.get("type") == "tool_result" for b in last["content"])
            )
            if not ok:
                raise ValueError(
                    "History malformed: expected a user turn of tool_result blocks "
                    "after tool_use. Check how the loop appends to messages."
                )
        if self._turn >= len(self._script):
            raise RuntimeError("Script exhausted — the stop condition failed.")
        resp = self._script[self._turn]
        self._turn += 1
        return resp


class MalformedMockClient(MockClient):
    """Rep-3 variant: second turn requests a tool that DOES NOT EXIST
    and omits a required param. the loop must survive: return an error
    string as the tool_result, keep the loop alive, reach end_turn.
    """

    def __init__(self):
        super().__init__()
        self._script[1] = MockResponse(
            content=[
                {"type": "text", "text": "Checking flood data."},
                {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "flood_risk",  # not in the registry
                    "input": {},  # missing params too
                },
            ],
            stop_reason="tool_use",
        )
        self._script[2] = MockResponse(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Flood data unavailable (tool error). Proceeding on listing "
                        "data alone: $415,000 asking, no comp verification possible. "
                        "Verdict: insufficient data — route to human analyst."
                    ),
                }
            ],
            stop_reason="end_turn",
        )
