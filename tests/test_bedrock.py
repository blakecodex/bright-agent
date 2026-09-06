"""the bedrock adapter's translation layer, tested without aws.
the client itself needs live credentials, so these pin the pure reshaping
functions in both directions - which is where the bugs would live."""
import unittest

from clients import to_converse, tool_config, from_converse


class ToConverse(unittest.TestCase):
    def test_question_becomes_one_text_block(self):
        out = to_converse([{"role": "user", "content": "Is 720 Shirley St fairly priced?"}])
        self.assertEqual(out, [{"role": "user", "content": [{"text": "Is 720 Shirley St fairly priced?"}]}])

    def test_tool_turns_map_field_for_field(self):
        history = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "comps next."},
                {"type": "tool_use", "id": "tu_2", "name": "comp_stats", "input": {"zip_code": "19130", "beds": 3}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_2", "content": "{\"comp_count\": 6}"},
            ]},
        ]
        out = to_converse(history)
        self.assertEqual(out[0]["content"][1]["toolUse"],
                         {"toolUseId": "tu_2", "name": "comp_stats", "input": {"zip_code": "19130", "beds": 3}})
        self.assertEqual(out[1]["content"][0]["toolResult"],
                         {"toolUseId": "tu_2", "content": [{"text": "{\"comp_count\": 6}"}]})

    def test_empty_text_blocks_are_dropped(self):
        out = to_converse([{"role": "assistant", "content": [{"type": "text", "text": ""},
                                                             {"type": "tool_use", "id": "t1", "name": "x", "input": {}}]}])
        self.assertEqual(len(out[0]["content"]), 1)   # converse rejects empty text


class ToolConfig(unittest.TestCase):
    def test_schema_moves_into_toolspec(self):
        tools = [{"name": "lookup_listing", "description": "fetch a record",
                  "input_schema": {"type": "object", "properties": {"address": {"type": "string"}},
                                   "required": ["address"]}}]
        cfg = tool_config(tools)
        spec = cfg["tools"][0]["toolSpec"]
        self.assertEqual(spec["name"], "lookup_listing")
        self.assertEqual(spec["inputSchema"]["json"]["required"], ["address"])


class FromConverse(unittest.TestCase):
    def test_tool_use_reply(self):
        resp = {"stopReason": "tool_use", "output": {"message": {"content": [
            {"text": "i need the record."},
            {"toolUse": {"toolUseId": "t1", "name": "lookup_listing", "input": {"address": "720 Shirley St"}}},
        ]}}}
        r = from_converse(resp)
        self.assertEqual(r.stop_reason, "tool_use")
        self.assertEqual(r.content[1], {"type": "tool_use", "id": "t1", "name": "lookup_listing",
                                        "input": {"address": "720 Shirley St"}})

    def test_end_turn_and_max_tokens_both_stop(self):
        for reason in ("end_turn", "max_tokens"):
            resp = {"stopReason": reason, "output": {"message": {"content": [{"text": "done."}]}}}
            self.assertEqual(from_converse(resp).stop_reason, "end_turn")

    def test_usage_flows_into_the_trace_shape(self):
        # bedrock reports tokens per call; the loop logs them when present, so cost is observable per turn
        resp = {"stopReason": "end_turn", "usage": {"inputTokens": 812, "outputTokens": 96},
                "output": {"message": {"content": [{"text": "done."}]}}}
        r = from_converse(resp)
        self.assertEqual(r.usage, {"input_tokens": 812, "output_tokens": 96})

    def test_no_usage_stays_none(self):
        resp = {"stopReason": "end_turn", "output": {"message": {"content": [{"text": "done."}]}}}
        self.assertIsNone(from_converse(resp).usage)


if __name__ == "__main__":
    unittest.main()
