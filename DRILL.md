# DRILL — 3 blocks, 10-min breaks. Speak aloud the whole time.

## Block 1 (~50 min) — READ + WARM-UP
1. Read mock_client.py aloud. Then tools.py. Logos (what it claims),
   taxis (how it's ordered), ergon (what it does).
2. Implement comp_stats() in tools.py. Median by hand. ~10 lines.
3. Say the loop from memory BEFORE writing it:
   "call -> check stop_reason -> execute tools -> append assistant turn ->
   append tool_results -> repeat -> cap at 10."

## Block 2 (~50 min) — BUILD, NO TIMER
1. agent.py from the contract. No AI. Docs allowed (that's the real rule).
2. `python run.py` until the happy path prints the verdict.
3. Narrate every choice as you make it. If you can't say why, stop — that's
   the flaw. Fix, continue.

## Block 3 (~50 min) — BREAK IT
1. `python run.py --broken`. Your loop must survive the unknown tool:
   error string as tool_result, loop continues, reaches end_turn.
2. Delete agent.py. Rebuild from blank. This is rep 2.
3. Stop at first flaw, fix, next take. Log time-to-green for both reps.

## PASS BAR (today)
- Happy path green, broken path green, rep 2 faster than rep 1.
- You can say the whole loop in under 30 seconds, cold.

Next session: rep 3 timed, then port this skeleton into the walkthrough repo.
