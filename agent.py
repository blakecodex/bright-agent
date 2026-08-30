from tools import execute_tool, TOOL_SCHEMAS

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

def run_agent(client, user_query):
    messages = [{"role": "user", "content": user_query}]

    for _ in range(10):
        response = client.create(messages=messages) #whole convo goes to the model; reply comes back as 'response'.

        if response.stop_reason == "end_turn": 
            return response.content[0]["text"]
        elif response.stop_reason == "tool_use":
            block = next(b for b in response.content if b["type"] == "tool_use")
            result = execute_tool(block["name"], block["input"])

            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": str(result),
                }],
            })

    return "iteration cap hit - no final answer"
            


