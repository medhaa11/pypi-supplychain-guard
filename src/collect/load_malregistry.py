import collections
import datetime as dt
import time
import zipfile

import duckdb
import pandas as pd

from src.features.extract import extract

ZIP = "data/raw/malregistry_dl/main.zip"
ROOT = "pypi_malregistry-main/"
EXTS = (".tar.gz", ".zip", ".whl")
MAX_MEMBER_BYTES = 50_000_000

zf = zipfile.ZipFile(ZIP)
members, other, odd = [], 0, []
for info in zf.infolist():
    if info.is_dir() or not info.filename.startswith(ROOT):
        continue
    parts = info.filename[len(ROOT):].split("/")
    if not parts[-1].endswith(EXTS):
        other += 1
        continue
    if len(parts) != 3:
        odd.append(info.filename)
        continue
    members.append((info, parts[0], parts[1], parts[2]))
print(f"archives found: {len(members)} | non-archive files: {other} | odd layout: {len(odd)}")
for o in odd[:5]:
    print("  odd:", o)

rows, errors, skipped_big, empty = [], collections.Counter(), 0, 0
t0 = time.time()
for i, (info, pkg, ver, fname) in enumerate(members, 1):
    if info.file_size > MAX_MEMBER_BYTES:
        skipped_big += 1
        continue
    try:
        feats = extract(fname, zf.read(info))
    except Exception as e:
        errors[type(e).__name__] += 1
        continue
    if feats is None:
        empty += 1
        continue
    rows.append({"archive": info.filename, "name": pkg, "version": ver,
                 "group": "malicious_ext", "label": 1, "source": "malregistry", **feats})
    if i % 1000 == 0:
        print(f"{i}/{len(members)} ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df["fmt"] = df["s_is_wheel"].map({1: "wheel", 0: "sdist"})
con = duckdb.connect("data/packages.duckdb")
con.execute("CREATE OR REPLACE TABLE malreg_features AS SELECT * FROM df")
df.to_parquet("data/malreg_features.parquet")
con.close()

print(f"\nprocessed: {len(df)} | empty: {empty} | failed: {sum(errors.values())} {dict(errors)} "
      f"| skipped >50MB: {skipped_big}")
print(df.fmt.value_counts().to_dict())
print("archives with no .py files:", int((df.s_n_py == 0).sum()),
      "| near-empty (<50 bytes of code):", int(((df.c_setup_bytes + df.c_pkg_bytes) < 50).sum()))
print("distinct package names:", df.name.nunique())

# is the timestamp inside the archives usable as a date proxy?
ts = df.x_mtime_max
print("\n=== timestamp inside archives ===")
print("missing/zero:", int((ts <= 0).sum()))
d = ts[ts > 0].map(lambda x: dt.datetime.fromtimestamp(x, dt.timezone.utc).date())
print("year counts:", d.map(lambda x: x.year).value_counts().sort_index().to_dict())
print("most common dates (a big spike means frozen/default timestamps):")
print(d.value_counts().head(5).to_string())
