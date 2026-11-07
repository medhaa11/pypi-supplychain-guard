import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import duckdb
import requests

TABLE = sys.argv[1] if len(sys.argv) > 1 else "benign_matched"
HEADERS = {"User-Agent": f"supplychain-research ({os.environ.get('SC_CONTACT', 'no-contact-set')})"}
OUT = pathlib.Path("data/raw/benign")
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect("data/packages.duckdb")
df = con.execute(f"SELECT * FROM {TABLE}").df()
rows = list(df.itertuples())


def get(r):
    dest = OUT / f"{r.name}__{r.version}__{r.filename}"
    if dest.exists():
        return str(dest)
    try:
        resp = requests.get(r.url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        time.sleep(0.1)
        return str(dest)
    except Exception as e:
        print("failed:", r.name, e, flush=True)
        return None


paths = []
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, p in enumerate(ex.map(get, rows), 1):
        paths.append(p)
        if i % 100 == 0:
            print(f"{i}/{len(rows)} done", flush=True)

df["archive_path"] = paths
con.execute(f"CREATE OR REPLACE TABLE {TABLE} AS SELECT * FROM df")
print("downloaded/present:", sum(p is not None for p in paths),
      "| failed:", sum(p is None for p in paths))
con.close()
