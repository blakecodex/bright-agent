"""
tools.py - the five things the assistant can do; the only file touching the data or the models.

two pieces the loop relies on:
- TOOLS_SCHEMAS     - the tool definitions the model sees (json schema)
- execute_tool      - one dispatch function: name and inputs go in, a dict comes out, and a failure comes back
                      as an error dict, never as an exception.

below that: look up a property, pull comps, read the market, 
search the notes, ask the price models for an estimate.
"""

import retrieval
from data import store
from ml import predict as ml_predict

# demo listing the mock client's script asks about (123 oak st, ellicott city md)
LISTINGS = {
    "123 Oak St": {
        "address": "123 Oak St",
        "zip_code": "21043",
        "beds": 3,
        "baths": 2,
        "sqft": 1650,
        "list_price": 415_000,
        "days_on_market": 12,
        "source": "fixture",
    }
}

RECENT_SALES = [
    {"zip_code": "21043", "beds": 3, "sale_price": 398_000},
    {"zip_code": "21043", "beds": 3, "sale_price": 407_000},
    {"zip_code": "21043", "beds": 3, "sale_price": 389_500},
    {"zip_code": "21043", "beds": 3, "sale_price": 421_000},
    {"zip_code": "21043", "beds": 4, "sale_price": 502_000},  # wrong bed count, gets filtered
]


def median(values):
    """sort, take the middle; average the two middles when even."""
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


# what the model is told it can call
TOOL_SCHEMAS = [
    {
        "name": "lookup_listing",
        "description": "fetch a property record by street address: beds, baths, sqft, year built, "
                       "last sale, assessed value. philadelphia addresses come from city records.",
        "input_schema": {
            "type": "object",
            "properties": {"address": {"type": "string"}},
            "required": ["address"],
        },
    },
    {
        "name": "comp_stats",
        "description": "recent comparable sales for a zip + bed count: count, median sale price, "
                       "median price per sqft, and the comps themselves (last 12 months).",
        "input_schema": {
            "type": "object",
            "properties": {
                "zip_code": {"type": "string"},
                "beds": {"type": "integer"},
                "months": {"type": "integer", "description": "look-back window, default 12"},
                "sqft": {"type": "integer", "description": "subject sqft, enables size-matched $/sqft"},
            },
            "required": ["zip_code", "beds"],
        },
    },
    {
        "name": "market_context",
        "description": "county-level market conditions from redfin: median days on market, months of "
                       "supply, inventory, regime (buyers/sellers/balanced) and the dom trend.",
        "input_schema": {
            "type": "object",
            "properties": {"region": {"type": "string", "description": "e.g. 'Philadelphia County, PA'"}},
            "required": [],
        },
    },
    {
        "name": "search_notes",
        "description": "retrieve short passages from the method notes (how verdicts are made, what the market "
                       "indicators mean, where the data comes from) to ground an explanation.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "predict_price",
        "description": "model-estimated sale price for a property record (hedonic regression + small "
                       "neural net trained on the same sales). input is the record from lookup_listing.",
        "input_schema": {
            "type": "object",
            "properties": {"property": {"type": "object"}},
            "required": ["property"],
        },
    },
]


def lookup_listing(address: str) -> dict:
    # demo listing first, then the real store, then an error dict
    rec = LISTINGS.get(address)
    if rec is not None:
        return dict(rec)
    row = store.find_property(address)
    if row is None:
        return {"error": f"no property record found for {address!r}"}
    # deed records have no asking price or days on market. use the last sale price
    # as the price in question; the caller can override both.
    return {
        "address": row["address"],
        "zip_code": row["zip"],
        "beds": row["beds"],
        "baths": row["baths"],
        "sqft": row["sqft"],
        "lot_sqft": row["lot_sqft"],
        "year_built": row["year_built"],
        "building": row["building"],
        "category": row["cat"],
        "quality_grade": row["quality_grade"],
        "ext_cond": row["ext_cond"],
        "central_air": row["central_air"],
        "last_sale_date": row["sale_date"],
        "last_sale_price": row["sale_price"],
        "assessed_value": row["market_value"],
        "list_price": row["sale_price"],
        "days_on_market": None,
        "lat": row["lat"],
        "lng": row["lng"],
        "source": "philadelphia opa via carto sql api",
    }


def comp_stats(zip_code: str, beds: int, months: int = 12, sqft: int = None) -> dict:
    zip_code = str(zip_code)
    beds = int(beds)
    real = store.comps(zip_code, beds, months=months, sqft=sqft)
    if real["comp_count"] > 0:
        real["comps"] = real["comps"][:8]  # keep the transcript small, stats carry the verdict
        real["source"] = "philadelphia opa sales"
        return real

    # fall back to the demo sales
    prices = [s["sale_price"] for s in RECENT_SALES if s["zip_code"] == zip_code and s["beds"] == beds]
    if not prices:
        return {"error": f"no comps found for zip {zip_code} with {beds} beds"}
    return {
        "zip_code": zip_code,
        "beds": beds,
        "comp_count": len(prices),
        "median_sale_price": median(prices),
        "median_ppsf": None,
        "window_months": 3,  # the demo sales are a 90-day window
        "source": "fixture",
    }


def market_context(region: str = "Philadelphia County, PA") -> dict:
    return store.market_context(region=region)


def predict_price(property: dict) -> dict:
    return ml_predict.predict(property)


def search_notes(query: str, k: int = 3) -> dict:
    return retrieval.search_notes(query, k=k)


TOOL_REGISTRY = {
    "lookup_listing": lookup_listing,
    "comp_stats": comp_stats,
    "market_context": market_context,
    "predict_price": predict_price,
    "search_notes": search_notes,
}


def execute_tool(name: str, tool_input: dict) -> dict:
    """dispatch one tool call. anything wrong comes back as an error dict, never an exception."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}
    if not isinstance(tool_input, dict):
        return {"error": f"bad input for {name!r}: expected an object"}
    try:
        return fn(**tool_input)
    except TypeError as e:
        return {"error": f"bad input for {name!r}: {e}"}
    except Exception as e:
        return {"error": f"{name!r} failed: {e}"}


if __name__ == "__main__":
    # quick smoke checks
    print(execute_tool("comp_stats", {"zip_code": "21043", "beds": 3}))
    print(execute_tool("lookup_listing", {"address": "3358 Livingston St"}))
    print({k: v for k, v in execute_tool("comp_stats", {"zip_code": "19134", "beds": 3}).items() if k != "comps"})
    print(execute_tool("market_context", {}))
    print(execute_tool("flood_risk", {}))  # unknown tool, comes back as an error