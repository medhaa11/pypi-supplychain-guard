import hashlib
import re
import time

import duckdb
import pandas as pd

from src.ecosystems.pypi_reader import read_archive


def h(b):
    return hashlib.sha1(re.sub(rb"\s+", b"", b)).hexdigest()


con = duckdb.connect("data/packages.duckdb")
meta = con.execute(
    'SELECT archive_path, name, "group" AS grp, fmt FROM features'
).df()

rows, t0 = [], time.time()
for i, r in enumerate(meta.itertuples(), 1):
    arc = read_archive(r.archive_path)
    py = {p: d for p, d in arc.files.items() if p.endswith(".py")}
    hashes = sorted(h(d) for d in py.values())
    rows.append({
        "archive_path": r.archive_path,
        "code_hash": hashlib.sha1("".join(hashes).encode()).hexdigest() if hashes else "NO_CODE",
        "setup_hash": h(py["setup.py"]) if "setup.py" in py else "",
        "name_stem": re.sub(r"\d+$", "", re.sub(r"[-_.]+", "-", r.name.lower())),
    })
    if i % 1000 == 0:
        print(f"{i}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)

fam = pd.DataFrame(rows)
con.execute("CREATE OR REPLACE TABLE families AS SELECT * FROM fam")
df = meta.merge(fam, on="archive_path")

print("\n=== duplicate structure (rows with code; NO_CODE excluded) ===")
for g, sub in df[df.code_hash != "NO_CODE"].groupby("grp"):
    sizes = sub.groupby("code_hash")["archive_path"].transform("size")
    print(f"{g}: rows={len(sub)} | unique code hashes={sub.code_hash.nunique()} "
          f"| % rows in a duplicate group={100 * (sizes > 1).mean():.1f} "
          f"| biggest group={sizes.max()}")
print("NO_CODE rows:", df[df.code_hash == "NO_CODE"].groupby("grp").size().to_dict())

m = df[(df.grp == "malicious") & (df.code_hash != "NO_CODE")]
b = df[(df.grp == "benign") & (df.code_hash != "NO_CODE")]
print("code hashes appearing in BOTH groups:", len(set(m.code_hash) & set(b.code_hash)))

print("\nmalicious name stems with most packages:")
print(m.name_stem.value_counts().head(8).to_string())
print("\nsetup.py hashes shared by 5+ malicious packages:",
      int((m[m.setup_hash != ""].setup_hash.value_counts() >= 5).sum()))
con.close()
