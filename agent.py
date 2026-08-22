"""
agent.py — YOU write this. Blank file is the drill.

CONTRACT
  run_agent(client, user_query: str) -> str
    - history = [{"role": "user", "content": user_query}]
    - loop:
        resp = client.create(messages=history, tools=TOOL_SCHEMAS)
        if resp.stop_reason == "end_turn": return the text
        else: for each tool_use block ->
            execute_tool(name, input)
            append assistant turn (resp.content) to history
            append user turn of tool_result blocks:
              {"type": "tool_result", "tool_use_id": <id>, "content": <str(result)>}
    - hard cap: max 10 iterations (runaway gate)

RUN
  python run.py            # happy path
  python run.py --broken   # malformed variant

Say it aloud while you type: what, why, which gate.
"""
