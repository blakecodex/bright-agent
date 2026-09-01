"""
tools.py - tool schemas + tool bodies. the only place the agent touches data or models.

the contract with the loop is small on purpose:
  TOOL_SCHEMAS  - what the model is told it can call (json schema, anthropic shape)
  execute_tool  - the single gate: name + input in, dict out, never an exception

everything below the gate is ordinary python a data scientist would write:
look a property up, pull comps, read the market, ask the model for a price.
"""

import retrieval
from data import store
from ml import predict as ml_predict

# ------------------------------------------------------------- fixtures
# the kit's demo listing lives on. the mock client asks for 123 oak st in 21043
# (ellicott city, md - also bright country) and we still answer it, from these.
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
    {"zip_code": "21043", "beds": 4, "sale_price": 502_000},  # filtered out by beds
]


def median(values):
    """median by hand: sort, split odd/even. no numpy, no statistics module - on purpose."""
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


# ------------------------------------------------------------- schemas
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


# ------------------------------------------------------------- bodies

def lookup_listing(address: str) -> dict:
    # real records first, fixture second, error last. the error is data, not an exception.
    rec = LISTINGS.get(address)
    if rec is not None:
        return dict(rec)
    row = store.find_property(address)
    if row is None:
        return {"error": f"no property record found for {address!r}"}
    # a recorded sale is not a listing: there is no asking price or dom. we surface the
    # last sale as the price on the table and let the caller override it.
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
        "list_price": row["sale_price"],   # default question: was this sale fairly priced?
        "days_on_market": None,            # not in public deed records
        "lat": row["lat"],
        "lng": row["lng"],
        "source": "philadelphia opa via carto sql api",
    }


def comp_stats(zip_code: str, beds: int, months: int = 12, sqft: int = None) -> dict:
    zip_code = str(zip_code)
    beds = int(beds)
    real = store.comps(zip_code, beds, months=months, sqft=sqft)
    if real["comp_count"] > 0:
        # trim the comp list for the transcript; the stats are what the verdict uses
        real["comps"] = real["comps"][:8]
        real["source"] = "philadelphia opa sales"
        return real

    # fixture path: same zip + beds filter, empty guard, median by hand
    prices = [s["sale_price"] for s in RECENT_SALES if s["zip_code"] == zip_code and s["beds"] == beds]
    if not prices:
        return {"error": f"no comps found for zip {zip_code} with {beds} beds"}
    return {
        "zip_code": zip_code,
        "beds": beds,
        "comp_count": len(prices),
        "median_sale_price": median(prices),
        "median_ppsf": None,
        "window_months": 3,   # the kit's story says "last 90 days"
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
    """single gate for all tool calls. unknown tool or bad input must NOT crash the
    agent - return an error dict instead. the loop hands it back to the model as data."""
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
    print(execute_tool("comp_stats", {"zip_code": "21043", "beds": 3}))          # fixture -> 402500.0, n=4
    print(execute_tool("lookup_listing", {"address": "3358 Livingston St"}))     # real record
    print({k: v for k, v in execute_tool("comp_stats", {"zip_code": "19134", "beds": 3}).items() if k != "comps"})
    print(execute_tool("market_context", {}))
    print(execute_tool("flood_risk", {}))                                          # error as data
