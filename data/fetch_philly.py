"""

fetch_philly.py - pull recorded sales from Philly's open-data api.
    - the endpoint takes a sql query over http and returns csv, no key needed.
    - not mls data - no asking prices or days on market - but real closed sales,
      which is what a comp is; pages land in cache/ as gzipped csv with a cheksum each.

      python -m data.fetch_philly                       # last 12 months, 500 rows per page
      python -m data.fetch_philly --since 2024-01-01    # deeper history

"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

from .store import CACHE_DIR

CARTO_SQL = "https://phl.carto.com/api/v2/sql"
TABLE = "opa_properties_public"

# the projection: one alias per column so the csv header is the schema store.py expects.
# casts happen server-side so the wire carries ints, not '1140.000'.
SELECT = """
SELECT parcel_number AS parcel,
       zip_code AS zip,
       CASE category_code_description WHEN 'SINGLE FAMILY' THEN 'SF' ELSE 'MF' END AS cat,
       building_code_description_new AS building,
       number_of_bedrooms AS beds,
       number_of_bathrooms AS baths,
       round(total_livable_area)::int AS sqft,
       round(total_area)::int AS lot_sqft,
       year_built,
       number_stories AS stories,
       exterior_condition AS ext_cond,
       interior_condition AS int_cond,
       quality_grade,
       central_air,
       garage_spaces::int AS garage,
       basements,
       fireplaces::int AS fireplaces,
       sale_date::date AS sale_date,
       sale_price::bigint AS sale_price,
       market_value::bigint AS market_value,
       location AS address,
       round(ST_Y(the_geom)::numeric, 4) AS lat,
       round(ST_X(the_geom)::numeric, 4) AS lng
FROM {table}
WHERE sale_price > 50000
  AND category_code_description IN ('SINGLE FAMILY', 'MULTI FAMILY')
  AND sale_date >= '{since}'
ORDER BY md5(parcel_number::text || sale_date::text)
LIMIT {limit} OFFSET {offset}
"""
# note the order by: a deterministic pseudo-random shuffle. any prefix of pages is
# then a fair sample of the whole window, so a partial pull is still usable.
# sale_price > 50000 drops the $1 family transfers and sheriff-sale noise.


def build_url(since, limit, offset, table=TABLE):
    sql = SELECT.format(table=table, since=since, limit=limit, offset=offset)
    return CARTO_SQL + "?" + urllib.parse.urlencode({"format": "csv", "q": " ".join(sql.split())})


def fetch_page(since, limit, offset, timeout=60):
    """one http round trip -> csv text. carto answers with a 200 and a header row even when empty."""
    url = build_url(since, limit, offset)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def count_rows(csv_text):
    # header + n data rows; carto uses \r\n
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    return max(0, len(lines) - 1)


def pull(since, page_size=500, max_pages=None, cache_dir=CACHE_DIR, sleep=0.5, verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    # clear old pages so a smaller refresh does not leave stale tails behind
    for name in os.listdir(cache_dir):
        if name.startswith("philly_sales_p") and name.endswith(".csv.gz"):
            os.remove(os.path.join(cache_dir, name))

    manifest = {"source": CARTO_SQL, "table": TABLE, "since": since, "page_size": page_size,
                "pulled_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "pages": []}
    page, total = 0, 0
    while True:
        if max_pages is not None and page >= max_pages:
            break
        text = fetch_page(since, page_size, page * page_size)
        n = count_rows(text)
        if n == 0:
            break
        path = os.path.join(cache_dir, f"philly_sales_p{page:03d}.csv.gz")
        payload = text.replace("\r\n", "\n").encode("utf-8")
        with gzip.open(path, "wb") as fh:
            fh.write(payload)
        manifest["pages"].append({"file": os.path.basename(path), "rows": n,
                                  "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()})
        total += n
        if verbose:
            print(f"page {page}: {n} rows -> {os.path.basename(path)}")
        page += 1
        if n < page_size:
            break
        time.sleep(sleep)  # be a polite guest on a public endpoint
    manifest["total_rows"] = total
    with open(os.path.join(cache_dir, "provenance_philly.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    if verbose:
        print(f"done: {total} rows in {page} pages")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="pull philadelphia sales into data/cache")
    default_since = (dt.date.today() - dt.timedelta(days=365)).isoformat()
    ap.add_argument("--since", default=default_since, help="earliest sale_date, yyyy-mm-dd")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--max-pages", type=int, default=None, help="stop early (smoke tests)")
    args = ap.parse_args(argv)
    try:
        pull(args.since, args.page_size, args.max_pages)
    except Exception as exc:  # network is the one dependency we do not control
        print(f"pull failed: {exc}\nthe cached snapshot in data/cache still works offline.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
