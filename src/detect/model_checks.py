import warnings

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

con = duckdb.connect("data/packages.duckdb")
df = con.execute("""
    SELECT x.*, s.family
    FROM features x JOIN split s USING (archive_path)
    WHERE s.split = 'train' AND x.fmt = 'sdist'
""").df().reset_index(drop=True)
con.close()


def tot(k):
    return df[f"c_setup_{k}"] + df[f"c_pkg_{k}"]


lines, code_bytes = tot("lines"), tot("bytes")
y, groups = df.label.values, df.family.values

size = pd.DataFrame({"lines": lines, "bytes": code_bytes, "files": tot("files"),
                     "n_files": df.s_n_files, "n_py": df.s_n_py})

rate_keys = ["exec_eval", "dyn_import", "proc", "net", "decode", "env", "sysinfo", "fs", "obf",
             "sens", "urls", "ips", "exfil", "hi_ent_str", "b64_blob", "long_str"]
D = pd.DataFrame({f"rate_{k}": tot(k) / (lines + 1) * 1000 for k in rate_keys})
for k in ["cmdclass", "cmd_subclass", "imp_proc", "imp_net", "proc", "net", "exec_eval", "decode"]:
    D[f"setup_{k}_flag"] = (df[f"c_setup_{k}"] > 0).astype(int)

raw = df[[c for c in df.columns if c.startswith("c_")]]


def model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                  random_state=0, class_weight="balanced_subsample")


def recall_at(yt, s, target):
    fpr, tpr, _ = roc_curve(yt, s)
    return tpr[fpr <= target].max()


def oof(X):
    X = np.asarray(X)
    out = np.zeros(len(X))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
        out[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


rows = []
scores = {}
for name, X in {"1 size only": size, "2 size-free rates": D, "A raw counts (earlier)": raw}.items():
    s = oof(X)
    scores[name] = s
    rows.append({"features": name, "AUC": round(roc_auc_score(y, s), 3),
                 "recall@1%FPR": round(recall_at(y, s, 0.01), 3),
                 "recall@5%FPR": round(recall_at(y, s, 0.05), 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== check 3: AUC inside size bands (model A raw counts / model 2 size-free) ===")
bands = pd.qcut(code_bytes, 5, duplicates="drop")
for b in sorted(bands.unique()):
    mk = (bands == b).values
    if len(set(y[mk])) < 2:
        continue
    print(f"{str(b):>22} n={int(mk.sum()):4d} mal={int(y[mk].sum()):4d} "
          f"AUC A={roc_auc_score(y[mk], scores['A raw counts (earlier)'][mk]):.3f} "
          f"AUC 2={roc_auc_score(y[mk], scores['2 size-free rates'][mk]):.3f}")

print("\n=== check 4: train on families first seen before 2024, test on 2024 families ===")
first = df.groupby("family")["date"].transform("min")
early, late = (first < "2024-01-01").values, (first >= "2024-01-01").values
print("early:", int(early.sum()), "rows,", int(y[early].sum()), "malicious | late:",
      int(late.sum()), "rows,", int(y[late].sum()), "malicious")
for name, X in {"2 size-free rates": D, "A raw counts": raw}.items():
    X = np.asarray(X)
    s = model().fit(X[early], y[early]).predict_proba(X[late])[:, 1]
    print(f"{name:>20}: AUC={roc_auc_score(y[late], s):.3f} "
          f"recall@5%FPR={recall_at(y[late], s, 0.05):.3f} "
          f"recall@1%FPR={recall_at(y[late], s, 0.01):.3f}")
