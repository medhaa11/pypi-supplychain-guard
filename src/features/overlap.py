import re
import time
import zipfile

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import HashingVectorizer

from src.ecosystems.pypi_reader import read_archive

ZIP = "data/raw/malregistry_dl/main.zip"
MAXCHARS, MINBYTES, THR, BLOCK = 30000, 50, 0.8, 500

con = duckdb.connect("data/packages.duckdb")
dd = con.execute("""SELECT archive_path, name, version, "group" AS grp,
                           c_setup_bytes + c_pkg_bytes AS code_bytes
                    FROM features ORDER BY archive_path""").df()
mr = con.execute("""SELECT archive, name, version, fmt, x_mtime_max,
                           c_setup_bytes + c_pkg_bytes AS code_bytes
                    FROM malreg_features ORDER BY archive""").df()
con.close()
print("datadog+benign rows:", len(dd), "| malregistry rows:", len(mr))

hv = HashingVectorizer(analyzer="char", ngram_range=(5, 5), n_features=2 ** 20, alternate_sign=False,
                       binary=True, norm="l2", lowercase=False, dtype=np.float32)


def code_text(arc):
    t = "\n".join(d.decode("utf-8", "ignore") for p, d in sorted(arc.files.items()) if p.endswith(".py"))
    return re.sub(r"\s+", " ", t)[:MAXCHARS]


def vectorize(get_text, n, label):
    mats, buf, t0 = [], [], time.time()
    for i in range(n):
        buf.append(get_text(i))
        if len(buf) == 1000 or i == n - 1:
            mats.append(hv.transform(buf).tocsr())
            buf = []
            print(f"{label}: {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    return sparse.vstack(mats).tocsr()


Xd = vectorize(lambda i: code_text(read_archive(dd.archive_path[i])), len(dd), "datadog+benign")

zf = zipfile.ZipFile(ZIP)
members = {info.filename: info for info in zf.infolist()}


def mr_text(i):
    a = mr.archive[i]
    return code_text(read_archive(a.rsplit("/", 1)[-1], zf.read(members[a])))


Xm = vectorize(mr_text, len(mr), "malregistry")
sparse.save_npz("data/X_dd.npz", Xd)
sparse.save_npz("data/X_mr.npz", Xm)
dd.to_parquet("data/X_dd_index.parquet")
mr.to_parquet("data/X_mr_index.parquet")

norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
dd["nn"] = dd.name.map(norm)
mr["nn"] = mr.name.map(norm)
mal_dd = dd[dd.grp == "malicious"]
print("\n=== name overlap with Datadog malicious ===")
print("malregistry rows whose package name is also in Datadog malicious:",
      int(mr.nn.isin(set(mal_dd.nn)).sum()), "of", len(mr))
kv = set(zip(mal_dd.nn, mal_dd.version.astype(str)))
print("same name AND version:", int(sum((a, str(b)) in kv for a, b in zip(mr.nn, mr.version))))

valid_dd = (dd.code_bytes >= MINBYTES).values
valid_mr = (mr.code_bytes >= MINBYTES).values
mal_cols = np.where(valid_dd & (dd.grp == "malicious").values)[0]
ben_cols = np.where(valid_dd & (dd.grp == "benign").values)[0]
XdM, XdB = Xd[mal_cols], Xd[ben_cols]
n = len(mr)
sim_mal, sim_ben = np.zeros(n, np.float32), np.zeros(n, np.float32)
er, ec = [], []
t0 = time.time()
for a in range(0, n, BLOCK):
    blk = Xm[a:a + BLOCK]
    sim_mal[a:a + BLOCK] = (blk @ XdM.T).toarray().max(axis=1)
    sim_ben[a:a + BLOCK] = (blk @ XdB.T).toarray().max(axis=1)
    S = (blk @ Xm.T).toarray()
    S[:, ~valid_mr] = 0
    S[~valid_mr[a:a + BLOCK], :] = 0
    r, c = np.nonzero(S >= THR)
    keep = c < a + r  # upper triangle only, keeps edge lists small
    er.append((a + r[keep]).astype(np.int32))
    ec.append(c[keep].astype(np.int32))
    if (a // BLOCK) % 5 == 0:
        print(f"similarity block {a}/{n} ({time.time() - t0:.0f}s)", flush=True)

er, ec = np.concatenate(er), np.concatenate(ec)
g = sparse.coo_matrix((np.ones(len(er), dtype=np.int8), (er, ec)), shape=(n, n))
ncomp, lab = connected_components(g, directed=False)
mr["cluster"], mr["sim_mal"], mr["sim_ben"], mr["valid"] = lab, sim_mal, sim_ben, valid_mr
mr["date"] = pd.to_datetime(mr.x_mtime_max, unit="s", utc=True, errors="coerce")
mr.loc[(mr.x_mtime_max <= 0) | (mr.date.dt.year < 2017), "date"] = pd.NaT
mr.drop(columns=["nn"]).to_parquet("data/malreg_clusters.parquet")

v = mr[mr.valid]
print(f"\n=== near-twins (cosine >= 0.8 / >= 0.95), {len(v)} packages with code ===")
print(f"to a Datadog MALICIOUS package: {100 * (v.sim_mal >= 0.8).mean():.1f}% / {100 * (v.sim_mal >= 0.95).mean():.1f}%")
print(f"to a BENIGN package:            {100 * (v.sim_ben >= 0.8).mean():.1f}% / {100 * (v.sim_ben >= 0.95).mean():.1f}%")
win = v[(v.date >= "2023-02-10") & (v.date <= "2023-03-10")]
rest = v.drop(win.index)
print(f"Feb-Mar 2023 window rows: {len(win)} -> share with Datadog twin >=0.8: {100 * (win.sim_mal >= 0.8).mean():.1f}%")
print(f"all other rows:           {len(rest)} -> share with Datadog twin >=0.8: {100 * (rest.sim_mal >= 0.8).mean():.1f}%")

cs = v.groupby("cluster").agg(size=("name", "size"), has_dd=("sim_mal", lambda s: (s >= 0.8).any()))
print(f"\n=== malregistry clusters at {THR} ===")
print("clusters:", len(cs), "| with a Datadog twin:", int(cs.has_dd.sum()),
      "| novel (no member has a Datadog twin):", int((~cs.has_dd).sum()))
print("rows in novel clusters:", int(cs[~cs.has_dd]["size"].sum()), "of", len(v))
print("largest clusters:")
for cid in cs["size"].sort_values(ascending=False).index[:8]:
    gr = v[v.cluster == cid]
    names = gr.name.str.lower().str.replace(r"[\d_.-]+$", "", regex=True).value_counts().head(3).to_dict()
    print(f"  size={len(gr):4d} dates={str(gr.date.min())[:10]}..{str(gr.date.max())[:10]} "
          f"median_bytes={int(gr.code_bytes.median()):6d} datadog_twin={100 * (gr.sim_mal >= 0.8).mean():.0f}% names={names}")
