"""
store.py - the local warehouse; sqlite, stdlib only.

two tables:
    - sales     - one row per recorded sale (Philly opa, ia the carto sql api)
    - market    - one row per county, month, and property type from redfin's tracker
two tables:
  sales   - one row per recorded sale (philadelphia opa, via the carto sql api)
  market  - one row per (county, month, property_type) from redfin's market tracker

tools never read csv files directly; they call this module and this module speaks sql.
the cache/ folder holds the gzipped pages exactly as the fetchers wrote them, so loading
from cache and loading from a fresh pull is one code path.
"""

import csv
import glob
import gzip
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
DB_PATH = os.path.join(CACHE_DIR, "bright.db")

# columns we keep from the opa extract, in the order the csv pages carry them.
# types matter for the mat later, so they're declared here.
SALES_COLUMNS = [
    ("parcel", "TEXT"), ("zip", "TEXT"), ("cat", "TEXT"), ("building", "TEXT"),
    ("beds", "INTEGER"), ("baths", "INTEGER"), ("sqft", "INTEGER"), ("lot_sqft", "INTEGER"),
    ("year_built", "INTEGER"), ("stories", "INTEGER"), ("ext_cond", "INTEGER"), ("int_cond", "INTEGER"),
    ("quality_grade", "TEXT"), ("central_air", "TEXT"), ("garage", "INTEGER"), ("basements", "TEXT"),
    ("fireplaces", "INTEGER"), ("sale_date", "TEXT"), ("sale_price", "INTEGER"), ("market_value", "INTEGER"),
    ("address", "TEXT"), ("lat", "REAL"), ("lng", "REAL"),
]

MARKET_COLUMNS = [
    ("period_begin", "TEXT"), ("region", "TEXT"), ("property_type", "TEXT"),
    ("median_sale_price", "REAL"), ("median_list_price", "REAL"), ("median_ppsf", "REAL"),
    ("homes_sold", "INTEGER"), ("pending_sales", "INTEGER"), ("new_listings", "INTEGER"),
    ("inventory", "INTEGER"), ("months_of_supply", "REAL"), ("median_dom", "REAL"),
    ("avg_sale_to_list", "REAL"), ("sold_above_list", "REAL"), ("price_drops", "REAL"),
]


def _coerce(value, sqltype):
    # empty string means the city didn't record it; store null, never zero.
    if value is None or value == "":
        return None
    try:
        if sqltype == "INTEGER":
            return int(float(value))
        if sqltype == "REAL":
            return float(value)
    except ValueError:
        return None
    return value


def _read_gz_table(path, delimiter=","):
    # yields dict rows from a gzipped delimited file. small files, so we just read them whole.
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        yield from csv.DictReader(fh, delimiter=delimiter)


def connect(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row  # rows behave like dicts; tools return plain dicts anyway
    # truncate the rollback journal instead of deleting it. synced folders (onedrive, dropbox)
    # and some sandboxes make file deletes slow or forbidden; a zero-length journal is just a write.
    con.execute("PRAGMA journal_mode=TRUNCATE")
    return con


def _has_tables(con):
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return {"sales", "market"} <= names


def build(db_path=DB_PATH, cache_dir=CACHE_DIR, verbose=True):
    """(re)build the sqlite file from whatever pages sit in cache/. idempotent."""
    con = connect(db_path)
    cur = con.cursor()

    # drop-and-recreate rather than deleting the file: works even where unlink is not allowed,
    # and a half-built file from an interrupted run is simply overwritten next time.
    cur.execute("DROP TABLE IF EXISTS sales")
    cur.execute("DROP TABLE IF EXISTS market")
    cols = ", ".join(f"{name} {typ}" for name, typ in SALES_COLUMNS)
    cur.execute(f"CREATE TABLE sales ({cols})")
    cols = ", ".join(f"{name} {typ}" for name, typ in MARKET_COLUMNS)
    cur.execute(f"CREATE TABLE market ({cols})")

    # sales pages: philly_sales_p000.csv.gz, p001, ... any number of them
    n_sales = 0
    for page in sorted(glob.glob(os.path.join(cache_dir, "philly_sales_p*.csv.gz"))):
        rows = []
        for r in _read_gz_table(page):
            rows.append(tuple(_coerce(r.get(name), typ) for name, typ in SALES_COLUMNS))
        marks = ",".join("?" * len(SALES_COLUMNS))
        cur.executemany(f"INSERT INTO sales VALUES ({marks})", rows)
        n_sales += len(rows)

    # redfin extract: one tsv, upper-case headers on the wire, lower-case in our table
    n_market = 0
    rf = os.path.join(cache_dir, "redfin_county_tracker.tsv.gz")
    if os.path.exists(rf):
        rows = []
        for r in _read_gz_table(rf, delimiter="\t"):
            rows.append(tuple(_coerce(r.get(name.upper()), typ) for name, typ in MARKET_COLUMNS))
        marks = ",".join("?" * len(MARKET_COLUMNS))
        cur.executemany(f"INSERT INTO market VALUES ({marks})", rows)
        n_market = len(rows)

    # the indexes the tools actually hit. zip+beds is the comps key, address is the lookup key.
    cur.execute("CREATE INDEX idx_sales_zip_beds ON sales (zip, beds)")
    cur.execute("CREATE INDEX idx_sales_address ON sales (address)")
    cur.execute("CREATE INDEX idx_market_region ON market (region, property_type, period_begin)")
    con.commit()
    if verbose:
        print(f"built {os.path.basename(db_path)}: {n_sales} sales, {n_market} market rows")
    return con


def ensure_db(db_path=DB_PATH):
    # lazy build so `python run.py` works on a fresh clone with only the cache present.
    # we check for the tables, not the file: an empty file left by a failed build must not
    # poison every later call with "no such table".
    con = connect(db_path)
    if not _has_tables(con):
        con.close()
        con = build(db_path, verbose=False)
    return con


# queries below:

def find_property(address, con=None):
    """exact-ish address match. opa addresses are upper-case with no punctuation."""
    con = con or ensure_db()
    tokens = " ".join(address.upper().replace(",", " ").replace(".", "").split()).split()
    # try the whole thing, then peel trailing tokens ("PHILADELPHIA", "PA", "19134", a unit) up to three times.
    # prefix match forgives a missing suffix; most recent sale wins when a house sold twice.
    for cut in range(len(tokens), max(1, len(tokens) - 3) - 1, -1):
        key = " ".join(tokens[:cut])
        row = con.execute("SELECT * FROM sales WHERE address = ? ORDER BY sale_date DESC LIMIT 1", (key,)).fetchone()
        if row is None:
            row = con.execute("SELECT * FROM sales WHERE address LIKE ? ORDER BY sale_date DESC LIMIT 1",
                              (key + "%",)).fetchone()
        if row is not None:
            return dict(row)
    return None


def comps(zip_code, beds, months=12, as_of=None, sqft=None, con=None):
    """
    comparable sales: same zip, same bed count, sold within the window.
        - returns the list of comps plus the stats the verdict needs 
        - the median is computed in sql with a window function
        - same idiom postgres and redshift use
        - because small homes carry a higher $/sqft by nature.
    """
    con = con or ensure_db()
    as_of = as_of or con.execute("SELECT max(sale_date) FROM sales").fetchone()[0]
    params = {"zip": str(zip_code), "beds": int(beds), "as_of": as_of, "months": -int(months)}

    rows = con.execute(
        """
        SELECT address, sale_date, sale_price, sqft, baths, building,
               CASE WHEN sqft > 0 THEN 1.0 * sale_price / sqft END AS ppsf
        FROM sales
        WHERE zip = :zip AND beds = :beds
          AND sale_date <= :as_of
          AND sale_date >  date(:as_of, :months || ' months')
        ORDER BY sale_date DESC
        """,
        params,
    ).fetchall()

    # median via row_number: middle row when odd, average of teh two middles when even
    # and it is the same idiom redshift/postgres people reach for.
    stats = con.execute(
        """
        WITH ranked AS (
            SELECT sale_price,
                   CASE WHEN sqft > 0 THEN 1.0 * sale_price / sqft END AS ppsf,
                   ROW_NUMBER() OVER (ORDER BY sale_price) AS rn,
                   COUNT(*)     OVER ()                    AS n
            FROM sales
            WHERE zip = :zip AND beds = :beds
              AND sale_date <= :as_of
              AND sale_date >  date(:as_of, :months || ' months')
        )
        SELECT n,
               AVG(sale_price) AS median_sale_price,
               (SELECT AVG(ppsf) FROM ranked WHERE ppsf IS NOT NULL) AS mean_ppsf
        FROM ranked
        WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
        """,
        params,
    ).fetchone()

    n = stats["n"] if stats and stats["n"] else 0
    out = {
        "zip_code": str(zip_code),
        "beds": int(beds),
        "window_months": months,
        "as_of": as_of,
        "comp_count": n,
        "median_sale_price": round(stats["median_sale_price"]) if n else None,
        "median_ppsf": None,
        "comps": [dict(r) for r in rows],
    }
    # ppsf medians in python, the list is already in memory
    out["median_ppsf"] = _median([r["ppsf"] for r in rows if r["ppsf"]])
    if sqft:
        similar = [r["ppsf"] for r in rows if r["ppsf"] and 0.7 * sqft <= (r["sqft"] or 0) <= 1.3 * sqft]
        out["similar_size_count"] = len(similar)
        out["median_ppsf_similar"] = _median(similar)
    return out


def _median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return round(values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2, 1)


def market_context(region="Philadelphia County, PA", property_type="All Residential", con=None):
    """
    latest *complete* month for a county from the redfin tracker, with a
    3-month trailing view so the verdict can say "cooling" or "heating".
    the most recent row is often a partial month (redfin publishes early), so
    we drop a trailing month whose homes_sold is far below the trailing median.
    """
    con = con or ensure_db()
    rows = con.execute(
        """
        SELECT period_begin, median_sale_price, median_dom, months_of_supply, inventory,
               homes_sold, avg_sale_to_list, sold_above_list, price_drops
        FROM market
        WHERE region = ? AND property_type = ?
        ORDER BY period_begin DESC
        LIMIT 7
        """,
        (region, property_type),
    ).fetchall()
    if not rows:
        return {"error": f"no market rows for {region!r} / {property_type!r}"}
    rows = [dict(r) for r in rows]
    sold = sorted(r["homes_sold"] or 0 for r in rows[1:])
    typical = sold[len(sold) // 2] if sold else 0
    if typical and (rows[0]["homes_sold"] or 0) < 0.5 * typical:
        rows = rows[1:]  # partial month, not evidence
    latest, prior = rows[0], rows[1:4]

    def avg(key):
        vals = [r[key] for r in prior if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    dom_prior = avg("median_dom")
    trend = "flat"
    if dom_prior and latest["median_dom"]:
        if latest["median_dom"] > dom_prior * 1.15:
            trend = "cooling"   # homes taking longer to sell
        elif latest["median_dom"] < dom_prior * 0.85:
            trend = "heating"
    mos = latest["months_of_supply"]
    # the old rule of thumb: under ~4 months favours sellers, over ~6 favours buyers
    regime = "balanced"
    if mos is not None:
        regime = "sellers" if mos < 4 else "buyers" if mos > 6 else "balanced"
    return {
        "region": region,
        "property_type": property_type,
        "period": latest["period_begin"],
        "median_sale_price": latest["median_sale_price"],
        "median_dom": latest["median_dom"],
        "months_of_supply": mos,
        "inventory": latest["inventory"],
        "homes_sold": latest["homes_sold"],
        "avg_sale_to_list": latest["avg_sale_to_list"],
        "dom_trend": trend,
        "regime": regime,
        "source": "redfin data center, county market tracker",
    }


def training_rows(con=None):
    """rows the ml layer can learn from: priced, sized, and not obviously a data error."""
    con = con or ensure_db()
    rows = con.execute(
        """
        SELECT * FROM sales
        WHERE sale_price BETWEEN 30000 AND 5000000
          AND sqft BETWEEN 300 AND 10000
          AND year_built BETWEEN 1700 AND 2026
        """
    ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    # smoke: build, then ask the three questions the agent asks
    con = build()
    print(find_property("3358 Livingston St"))
    c = comps("19134", 3, con=con)
    print({k: v for k, v in c.items() if k != "comps"})
    print(market_context(con=con))
