import re
import time

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import HashingVectorizer

from src.ecosystems.pypi_reader import read_archive

CUTOFF = "2025-01-01"
THR_FINAL = 0.8
MAXCHARS = 60000

con = duckdb.connect("data/packages.duckdb")
df = con.execute("""
    SELECT x.archive_path, x.name, x."group" AS grp, x.label, x.fmt, x.date,
           s.family AS old_family, s.split AS old_split
    FROM features x JOIN split s USING (archive_path)
    WHERE s.split IN ('train', 'test')
""").df().reset_index(drop=True)
triv = con.execute("SELECT archive_path FROM split WHERE split = 'trivial'").df()
n = len(df)
print("modeling rows:", n)


def code_text(path):
    arc = read_archive(path)
    t = "\n".join(d.decode("utf-8", "ignore")
                  for p, d in sorted(arc.files.items()) if p.endswith(".py"))
    return re.sub(r"\s+", " ", t)[:MAXCHARS]


t0 = time.time()
texts = []
for i, p in enumerate(df.archive_path, 1):
    texts.append(code_text(p))
    if i % 1000 == 0:
        print(f"read {i}/{n} ({time.time() - t0:.0f}s)", flush=True)

hv = HashingVectorizer(analyzer="char", ngram_range=(5, 5), n_features=2 ** 20,
                       alternate_sign=False, binary=True, norm="l2", lowercase=False)
X = hv.transform(texts).tocsr()
S = np.zeros((n, n), dtype=np.float32)
for a in range(0, n, 500):
    S[a:a + 500] = (X[a:a + 500] @ X.T).toarray()
np.fill_diagonal(S, 0)
print(f"similarity matrix done ({time.time() - t0:.0f}s)")

tr = np.where(df.old_split == "train")[0]
te = np.where(df.old_split == "test")[0]
fam = df.old_family.values


def twin_report(train_idx, test_idx, label):
    ms = S[np.ix_(test_idx, train_idx)].max(axis=1)
    g = df.grp.values[test_idx]
    for grp in ("malicious", "benign"):
        v = ms[g == grp]
        print(f"  {label} {grp}: test rows={len(v)} | median max-sim to train={np.median(v):.2f} "
              f"| share >=0.8: {100 * (v >= 0.8).mean():.1f}% | >=0.95: {100 * (v >= 0.95).mean():.1f}%")


print("\n=== OLD split: test packages with a near-twin in train ===")
twin_report(tr, te, "old")

M = S[np.ix_(tr, tr)].copy()
M[fam[tr][:, None] == fam[tr][None, :]] = 0
mx = M.max(axis=1)
for grp in ("malicious", "benign"):
    v = mx[df.grp.values[tr] == grp]
    print(f"train rows with a >=0.8 twin in a DIFFERENT old family ({grp}): {100 * (v >= 0.8).mean():.1f}%")

old_first = pd.Series(np.arange(n)).groupby(df.old_family.values).transform("min").values


def clusters(thr):
    r, c = np.where(S >= thr)
    rows = np.concatenate([r, np.arange(n)])
    cols = np.concatenate([c, old_first])
    g = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return connected_components(g, directed=False)[1]


print("\n=== cluster sensitivity (old families + similarity links) ===")
print("old families:", df.old_family.nunique())
labels = {}
for thr in (0.7, 0.8, 0.9):
    lab = clusters(thr)
    labels[thr] = lab
    sizes = np.sort(np.bincount(lab))[::-1]
    print(f"thr {thr}: clusters={len(sizes)} | largest={sizes[:6].tolist()}")

lab = labels[THR_FINAL]
df["cluster"] = lab
first = df.groupby("cluster")["date"].transform("min")
df["split2"] = np.where(first >= CUTOFF, "test", "train")

print(f"\n=== NEW split (threshold {THR_FINAL}) ===")
print(df.groupby(["split2", "grp"]).size().unstack(fill_value=0))
print(df.groupby(["split2", "fmt"]).size().unstack(fill_value=0))
print("test share:", round(100 * (df.split2 == "test").mean(), 1), "%")
print("rows whose side changed vs old split:", int((df.split2 != df.old_split).sum()))
twin_report(np.where(df.split2 == "train")[0], np.where(df.split2 == "test")[0], "new")

out = pd.concat([
    df[["archive_path", "cluster", "split2"]].rename(columns={"cluster": "family", "split2": "split"}),
    triv.assign(family=-1, split="trivial"),
])
con.execute("CREATE OR REPLACE TABLE split2 AS SELECT * FROM out")
out.to_parquet("data/split2.parquet")
con.close()
