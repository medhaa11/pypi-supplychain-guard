import pathlib
import re

import duckdb
import pandas as pd

ROOT = pathlib.Path("data/raw/datadog/samples/pypi")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")

rows, odd = [], []
for z in ROOT.rglob("*.zip"):
    parts = z.relative_to(ROOT).parts
    if len(parts) == 4:
        category, name, version, fname = parts
    elif len(parts) == 3:
        category, name, fname = parts
        version = None  # early samples: no version folder
    else:
        odd.append(str(z))
        continue
    m = DATE_RE.match(fname)
    rows.append({
        "name": name,
        "version": version,
        "has_version": version is not None,
        "label": 1,
        "source": "datadog",
        "category": category,
        "date_added": m.group(1) if m else None,
        "archive_path": str(z),
    })

df = pd.DataFrame(rows)
print("rows loaded:", len(df))
print("truly unexpected layouts:", len(odd))
print(df["category"].value_counts())
print("with version:", int(df["has_version"].sum()),
      "| without version:", int((~df["has_version"]).sum()))
print("date range:", df["date_added"].min(), "to", df["date_added"].max())
print("missing dates:", int(df["date_added"].isna().sum()))
print("duplicate archive_path:", int(df.duplicated("archive_path").sum()))
print("duplicate (name, version) among versioned:",
      int(df[df["has_version"]].duplicated(["name", "version"]).sum()))

con = duckdb.connect("data/packages.duckdb")
con.execute("CREATE OR REPLACE TABLE malicious AS SELECT * FROM df")
print("rows in table:", con.execute("SELECT COUNT(*) FROM malicious").fetchone()[0])
con.close()
