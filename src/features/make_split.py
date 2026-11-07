import duckdb
import pandas as pd

CUTOFF = "2025-01-01"
MIN_BYTES = 50  # packages with less Python code than this carry no code signal

con = duckdb.connect("data/packages.duckdb")
df = con.execute("""
    SELECT f.archive_path, f.name, f."group" AS grp, f.label, f.fmt, f.date,
           f.c_setup_bytes + f.c_pkg_bytes AS code_bytes,
           fam.code_hash, fam.name_stem
    FROM features f JOIN families fam USING (archive_path)
    WHERE fam.code_hash <> 'NO_CODE'
""").df()

df["trivial"] = df.code_bytes < MIN_BYTES
print("rows with some .py code:", len(df))
print("near-empty (<%d bytes) by group:" % MIN_BYTES, df[df.trivial].groupby("grp").size().to_dict())
mid = df[(df.code_bytes >= MIN_BYTES) & (df.code_bytes < 200)]
print("sensitivity: rows with 50-199 bytes by group:", mid.groupby("grp").size().to_dict())

triv = df[df.trivial].copy()
df = df[~df.trivial].reset_index(drop=True)

# union-find: link rows sharing a name stem or a code hash
parent = list(range(len(df)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra


for key in ("name_stem", "code_hash"):
    first = {}
    for i, v in enumerate(df[key]):
        if v in first:
            union(first[v], i)
        else:
            first[v] = i

df["family"] = [find(i) for i in range(len(df))]
fam_sizes = df.groupby("family").size()
print("\nmodeling rows:", len(df), "| families:", len(fam_sizes))
print("largest families (rows):", sorted(fam_sizes.tolist(), reverse=True)[:8])

first_date = df.groupby("family")["date"].transform("min")
last_date = df.groupby("family")["date"].transform("max")
straddle = df[(first_date < CUTOFF) & (last_date >= CUTOFF)]
print("families spanning the cutoff:", straddle.family.nunique(),
      "| rows a naive date split would have misplaced:", int((straddle.date >= CUTOFF).sum()))

df["split"] = (first_date >= CUTOFF).map({True: "test", False: "train"})
print("\n=== split sizes ===")
print(df.groupby(["split", "grp"]).size().unstack(fill_value=0))
print(df.groupby(["split", "fmt"]).size().unstack(fill_value=0))
print("test share of rows:", round((df.split == "test").mean() * 100, 1), "%")

triv["family"], triv["split"] = -1, "trivial"
out = pd.concat([df, triv])[["archive_path", "family", "split"]]
con.execute("CREATE OR REPLACE TABLE split AS SELECT * FROM out")
out.to_parquet("data/split.parquet")
con.close()
