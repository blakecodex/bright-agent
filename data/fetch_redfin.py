"""
fetch_redfin.py - stream redfin's county market tracker and keep the bright-mls
footprint counties. stdlib only, and it never holds the file in memory.

redfin publishes the whole country as one gzipped tsv (~240 mb compressed,
~700 mb inflated). we open the http response, wrap it in a gzip reader, and walk
it line by line - decompress, test, keep or drop. that is the streaming idea in
its simplest form: constant memory regardless of file size.

    python -m data.fetch_redfin              # ~2-4 minutes on a home connection
    python -m data.fetch_redfin --since 2022-01-01

output: cache/redfin_county_tracker.tsv.gz with the columns store.py expects.
the committed snapshot was produced by this filter on 2026-09-01.
"""

import argparse
import datetime as dt
import gzip
import io
import json
import os
import sys
import urllib.request

from .store import CACHE_DIR

URL = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/county_market_tracker.tsv000.gz"

# bright mls covers dc, md, va, de, nj (south), pa (east), wv - these are the busy counties
KEEP_REGIONS = {
    "Philadelphia County, PA", "Montgomery County, PA", "Delaware County, PA", "Bucks County, PA",
    "Chester County, PA", "Camden County, NJ", "Burlington County, NJ", "Gloucester County, NJ",
    "New Castle County, DE", "Baltimore County, MD", "Montgomery County, MD", "Fairfax County, VA",
    "District of Columbia, DC",
}
KEEP_TYPES = {"All Residential", "Single Family Residential"}
# heads-up for philadelphia: redfin files rowhomes under "Townhouse", so "single family"
# there is a small detached-only slice. use "All Residential" for the city.

OUT_COLUMNS = ["PERIOD_BEGIN", "REGION", "PROPERTY_TYPE", "MEDIAN_SALE_PRICE", "MEDIAN_LIST_PRICE",
               "MEDIAN_PPSF", "HOMES_SOLD", "PENDING_SALES", "NEW_LISTINGS", "INVENTORY",
               "MONTHS_OF_SUPPLY", "MEDIAN_DOM", "AVG_SALE_TO_LIST", "SOLD_ABOVE_LIST", "PRICE_DROPS"]


def _tidy(value):
    # redfin wraps everything in quotes and carries 12-digit floats; trim both
    value = value.strip('"')
    if value == "":
        return ""
    try:
        num = float(value)
    except ValueError:
        return value
    return str(round(num)) if abs(num) >= 100 else str(round(num, 3))


def filter_lines(lines, since, keep_regions=KEEP_REGIONS, keep_types=KEEP_TYPES):
    """
    pure function over an iterable of tsv lines (header first). yields kept
    output rows as lists of strings. separated from the http bit so it can be
    tested against the cached file without touching the network.
    """
    header = None
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if header is None:
            header = [p.strip('"') for p in parts]
            idx = {name: header.index(name) for name in OUT_COLUMNS + ["PERIOD_DURATION"]}
            yield OUT_COLUMNS
            continue
        if len(parts) < len(header):
            continue
        region = parts[idx["REGION"]].strip('"')
        if region not in keep_regions:
            continue
        if parts[idx["PROPERTY_TYPE"]].strip('"') not in keep_types:
            continue
        if parts[idx["PERIOD_DURATION"]].strip('"') != "30":   # monthly rows only
            continue
        if parts[idx["PERIOD_BEGIN"]].strip('"') < since:
            continue
        yield [parts[idx[c]].strip('"') if c in ("PERIOD_BEGIN", "REGION", "PROPERTY_TYPE")
               else _tidy(parts[idx[c]]) for c in OUT_COLUMNS]


def stream(url=URL, timeout=120):
    """yield decoded lines from the remote gzip without downloading it first."""
    resp = urllib.request.urlopen(url, timeout=timeout)
    with gzip.GzipFile(fileobj=resp) as gz:
        for raw in io.TextIOWrapper(gz, encoding="utf-8"):
            yield raw


def pull(since, cache_dir=CACHE_DIR, verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, "redfin_county_tracker.tsv.gz")
    body = []
    for i, row in enumerate(filter_lines(stream(), since)):
        if i == 0:
            continue  # header; we write our own below
        body.append("\t".join(row))
        if verbose and len(body) % 200 == 0:
            print(f"  kept {len(body)} rows so far...")
    body.sort(key=lambda s: s.split("\t")[:3])  # (period, region, type): stable, diff-friendly
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(OUT_COLUMNS) + "\n" + "\n".join(body) + "\n")
    manifest = {"source": URL, "since": since, "rows": len(body),
                "pulled_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "regions": sorted(KEEP_REGIONS), "property_types": sorted(KEEP_TYPES)}
    with open(os.path.join(cache_dir, "provenance_redfin.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    if verbose:
        print(f"done: {len(body)} rows -> {os.path.basename(out_path)}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="stream redfin county tracker into data/cache")
    ap.add_argument("--since", default="2023-01-01")
    args = ap.parse_args(argv)
    try:
        pull(args.since)
    except Exception as exc:
        print(f"pull failed: {exc}\nthe cached extract in data/cache still works offline.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
