import warnings

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from src.detect.rules import apply_rules

warnings.filterwarnings("ignore")

con = duckdb.connect("data/packages.duckdb")
df = con.execute("""
    SELECT x.*, s.family AS cluster
    FROM features x JOIN split2 s USING (archive_path)
    WHERE s.split = 'train' AND x.fmt = 'sdist'
""").df().reset_index(drop=True)
con.close()

y, groups = df.label.values, df.cluster.values
csize = df.groupby("cluster")["cluster"].transform("size").values
w = 1.0 / csize  # each cluster counts once
print("sdist train rows:", len(df), "| clusters:", df.cluster.nunique())
for lab, name in ((1, "malicious"), (0, "benign")):
    print(f"  {name}: rows={int((y == lab).sum())} | distinct clusters={df.cluster[y == lab].nunique()}")


def tot(k):
    return df[f"c_setup_{k}"] + df[f"c_pkg_{k}"]


lines = tot("lines")
size = pd.DataFrame({"lines": lines, "bytes": tot("bytes"), "files": tot("files"),
                     "n_files": df.s_n_files, "n_py": df.s_n_py})
raw = df[[c for c in df.columns if c.startswith("c_")]]
keys = ["exec_eval", "dyn_import", "proc", "net", "decode", "env", "sysinfo", "fs", "obf",
        "sens", "urls", "ips", "exfil", "hi_ent_str", "b64_blob", "long_str"]
rates = pd.DataFrame({f"rate_{k}": tot(k) / (lines + 1) * 1000 for k in keys})
for k in ["cmdclass", "cmd_subclass", "imp_proc", "imp_net", "proc", "net", "exec_eval", "decode"]:
    rates[f"setup_{k}_flag"] = (df[f"c_setup_{k}"] > 0).astype(int)

configs = {
    "1 size only": size,
    "2 behavior counts": raw,
    "3 behavior rates": rates,
    "4 size + behavior": pd.concat([raw, size.add_prefix("sz_")], axis=1),
}


def model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                  random_state=0, class_weight="balanced_subsample")


def recall_at(yt, s, target, sw=None):
    fpr, tpr, _ = roc_curve(yt, s, sample_weight=sw)
    return tpr[fpr <= target].max()


def oof(X):
    X = np.asarray(X)
    out = np.zeros(len(X))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
        out[te] = model().fit(X[tr], y[tr], sample_weight=w[tr]).predict_proba(X[te])[:, 1]
    return out


rows, scores = [], {}
for name, X in configs.items():
    s = oof(X)
    scores[name] = s
    rows.append({"features": name,
                 "AUC": round(roc_auc_score(y, s), 3),
                 "AUC_cluster_wtd": round(roc_auc_score(y, s, sample_weight=w), 3),
                 "recall@1%FPR_wtd": round(recall_at(y, s, 0.01, w), 3),
                 "recall@5%FPR_wtd": round(recall_at(y, s, 0.05, w), 3)})
print()
print(pd.DataFrame(rows).to_string(index=False))

R = apply_rules(df)
hit = R["any"].values
print("\nrule baseline (same rows): recall=%.3f (cluster-wtd %.3f) at FPR=%.3f (cluster-wtd %.3f)" % (
    hit[y == 1].mean(), np.average(hit[y == 1], weights=w[y == 1]),
    hit[y == 0].mean(), np.average(hit[y == 0], weights=w[y == 0])))

print("\n=== AUC within size bands (cluster-weighted): size-only vs size+behavior ===")
bands = pd.qcut(tot("bytes"), 5, duplicates="drop")
for b in sorted(bands.unique()):
    mk = (bands == b).values
    if len(set(y[mk])) < 2:
        continue
    a1 = roc_auc_score(y[mk], scores["1 size only"][mk], sample_weight=w[mk])
    a4 = roc_auc_score(y[mk], scores["4 size + behavior"][mk], sample_weight=w[mk])
    print(f"{str(b):>24} n={int(mk.sum()):4d} mal={int(y[mk].sum()):4d} "
          f"benign={int((y[mk] == 0).sum()):4d} | size-only={a1:.3f}  size+behavior={a4:.3f}")
