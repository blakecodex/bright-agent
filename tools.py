"""
tools.py — tool schemas + bodies. Stdlib only.
Warm-up task: implement comp_stats() yourself (median by hand, no numpy).
"""

LISTINGS = {
    "123 Oak St": {
        "address": "123 Oak St",
        "zip_code": "21043",
        "beds": 3,
        "baths": 2,
        "sqft": 1650,
        "list_price": 415_000,
        "days_on_market": 12,
    }
}

RECENT_SALES = [
    {"zip_code": "21043", "beds": 3, "sale_price": 398_000},
    {"zip_code": "21043", "beds": 3, "sale_price": 407_000},
    {"zip_code": "21043", "beds": 3, "sale_price": 389_500},
    {"zip_code": "21043", "beds": 3, "sale_price": 421_000},
    {"zip_code": "21043", "beds": 4, "sale_price": 502_000},  # filtered out by beds
]

TOOL_SCHEMAS = [
    {
        "name": "lookup_listing",
        "description": "Fetch a listing record by street address.",
        "input_schema": {
            "type": "object",
            "properties": {"address": {"type": "string"}},
            "required": ["address"],
        },
    },
    {
        "name": "comp_stats",
        "description": "Median sale price of recent comparable sales (zip + beds).",
        "input_schema": {
            "type": "object",
            "properties": {
                "zip_code": {"type": "string"},
                "beds": {"type": "integer"},
            },
            "required": ["zip_code", "beds"],
        },
    },
]


def lookup_listing(address: str) -> dict:
    rec = LISTINGS.get(address)
    if rec is None:
        return {"error": f"no listing found for {address!r}"}
    return rec


def comp_stats(zip_code: str, beds: int) -> dict:
    # TODO(BLAKE): implement.
    # 1. filter RECENT_SALES by zip_code AND beds
    # 2. median of sale_price (sort; even n -> mean of middle two)
    # 3. return {"median": ..., "n": ...}  ; if n == 0 return {"error": "..."}
    raise NotImplementedError


TOOL_REGISTRY = {
    "lookup_listing": lookup_listing,
    "comp_stats": comp_stats,
}


def execute_tool(name: str, tool_input: dict) -> dict:
    """Single gate for all tool calls. Unknown tool or bad input must NOT
    crash the agent — return an error dict instead."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return fn(**tool_input)
    except TypeError as e:
        return {"error": f"bad input for {name!r}: {e}"}
    except Exception as e:
        return {"error": f"{name!r} failed: {e}"}
