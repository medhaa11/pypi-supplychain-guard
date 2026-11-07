import os
import random
import re
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import duckdb
import pandas as pd
import requests

SEED = 42
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MAX_SCAN = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
MAX_BYTES = 5 * 1024 * 1024
WHEEL_PCT = 15  # ~15% of malicious_intent archives are unpacked wheels
HEADERS = {"User-Agent": f"supplychain-research ({os.environ.get('SC_CONTACT', 'no-contact-set')})"}


def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


con = duckdb.connect("data/packages.duckdb")
lo, hi = con.execute("SELECT MIN(date_added), MAX(date_added) FROM malicious").fetchone()
mal_names = {norm(r[0]) for r in con.execute("SELECT DISTINCT name FROM malicious").fetchall()}
year_counts = con.execute("""
    SELECT substr(date_added, 1, 4), COUNT(*) FROM malicious
    WHERE category = 'malicious_intent' GROUP BY 1
""").fetchall()
total = sum(c for _, c in year_counts)
quota = {y: round(TARGET * c / total) for y, c in year_counts}
print("window:", lo, "to", hi)
print("quota per year:", quota)

r = requests.get(
    "https://pypi.org/simple/",
    headers={**HEADERS, "Accept": "application/vnd.pypi.simple.v1+json"},
    timeout=120,
)
names = [p["name"] for p in r.json()["projects"] if norm(p["name"]) not in mal_names]
rng = random.Random(SEED)
rng.shuffle(names)
names = names[:MAX_SCAN]


def fetch(name):
    time.sleep(0.1)
    try:
        resp = requests.get(f"https://pypi.org/pypi/{name}/json", headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return []
        want_wheel = zlib.crc32(name.encode()) % 100 < WHEEL_PCT
        out = []
        for ver, files in resp.json().get("releases", {}).items():
            for f in files:
                if f.get("yanked") or f.get("size", 0) > MAX_BYTES:
                    continue
                ptype = f.get("packagetype")
                if want_wheel:
                    if ptype != "bdist_wheel" or not f["filename"].endswith("-none-any.whl"):
                        continue
                elif ptype != "sdist":
                    continue
                t = (f.get("upload_time_iso_8601") or "")[:10]
                if lo <= t <= hi:
                    out.append({
                        "name": name, "version": ver, "label": 0,
                        "source": "pypi_matched", "upload_time": t,
                        "filename": f["filename"], "size": f["size"], "url": f["url"],
                    })
        return out
    except Exception:
        return []


filled, rows = Counter(), []
CHUNK = 200
with ThreadPoolExecutor(max_workers=4) as ex:
    for start in range(0, len(names), CHUNK):
        chunk = names[start:start + CHUNK]
        for cands in ex.map(fetch, chunk):
            open_c = [c for c in cands
                      if filled[c["upload_time"][:4]] < quota.get(c["upload_time"][:4], 0)]
            if open_c:
                c = rng.choice(open_c)
                filled[c["upload_time"][:4]] += 1
                rows.append(c)
        print(f"scanned {min(start + CHUNK, len(names))}, kept {len(rows)}", flush=True)
        if all(filled[y] >= q for y, q in quota.items()):
            break

df = pd.DataFrame(rows)
if len(df) == 0:
    raise SystemExit("nothing kept")
df["fmt"] = df["filename"].str.endswith(".whl").map({True: "wheel", False: "sdist"})
print("kept:", len(df), "of target", TARGET)
print("filled vs quota:", {y: (filled[y], quota[y]) for y in sorted(quota)})
print(df["fmt"].value_counts())
print("size (bytes) median:", int(df["size"].median()), "| max:", int(df["size"].max()))
con.execute("CREATE OR REPLACE TABLE benign_matched AS SELECT * FROM df")
con.close()
