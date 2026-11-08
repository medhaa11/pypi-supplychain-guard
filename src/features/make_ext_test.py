import re

import duckdb
import numpy as np
import pandas as pd

mr = pd.read_parquet("data/malreg_clusters.parquet")
con = duckdb.connect("data/packages.duckdb")
dd = con.execute("""SELECT name, version FROM features WHERE "group" = 'malicious'""").df()

norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
dd_keys = {(norm(a), str(b)) for a, b in zip(dd.name, dd.version) if b is not None}
mr["exact_dup"] = [(norm(a), str(b)) in dd_keys for a, b in zip(mr.name, mr.version)]

v = mr[mr.valid]
cluster_has_twin = v.groupby("cluster").sim_mal.max() >= 0.8
mr["novel"] = mr.valid & ~mr.cluster.map(cluster_has_twin).fillna(True).astype(bool)
keep = mr[mr.novel & ~mr.exact_dup].copy()
keep["w"] = 1.0 / keep.groupby("cluster")["cluster"].transform("size")

print("rows with code:", int(mr.valid.sum()), "| in novel clusters:", int(mr.novel.sum()),
      "| after removing exact name+version matches:", len(keep))
print("\nfmt: rows / clusters")
for fmt, g in keep.groupby("fmt"):
    print(f"  {fmt}: {len(g)} rows / {g.cluster.nunique()} clusters")
sz = keep.groupby("cluster").size()
print("\ncluster size: singletons=%d | 2-5=%d | 6-20=%d | >20=%d | largest=%d" % (
    (sz == 1).sum(), ((sz >= 2) & (sz <= 5)).sum(), ((sz >= 6) & (sz <= 20)).sum(),
    (sz > 20).sum(), sz.max()))
print("code bytes percentiles (10/25/50/75/90):",
      np.percentile(keep.code_bytes, [10, 25, 50, 75, 90]).astype(int).tolist())
print("share under 500 bytes of code: %.1f%%" % (100 * (keep.code_bytes < 500).mean()))
print("year (timestamp proxy):", keep.date.dt.year.value_counts().sort_index().to_dict())

strict = mr[mr.valid & (mr.sim_mal < 0.6)]
print(f"\nsensitivity: rows with NO Datadog neighbour above 0.6: {len(strict)} "
      f"in {strict.cluster.nunique()} clusters")

print("\nlargest kept clusters:")
for cid in sz.sort_values(ascending=False).index[:6]:
    g = keep[keep.cluster == cid]
    print(f"  size={len(g):4d} dates={str(g.date.min())[:10]}..{str(g.date.max())[:10]} "
          f"median_bytes={int(g.code_bytes.median())}")

out = keep[["archive", "name", "version", "fmt", "cluster", "w", "date", "code_bytes"]]
con.execute("CREATE OR REPLACE TABLE ext_test AS SELECT * FROM out")
out.to_parquet("data/ext_test.parquet")
con.close()
