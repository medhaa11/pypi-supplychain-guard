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
    SELECT x.*, s.family AS cluster
    FROM features x JOIN split2 s USING (archive_path)
    WHERE s.split = 'train' AND x.fmt = 'sdist'
""").df().reset_index(drop=True)
con.close()

y, groups = df.label.values, df.cluster.values
w = 1.0 / df.groupby("cluster")["cluster"].transform("size").values
SIZEISH = {"lines", "bytes", "files", "calls"}
base = [c for c in df.columns if c.startswith("c_") and "_top_" not in c
        and c.split("_", 2)[2] not in SIZEISH]
top = [f"c_{s}_top_{k}" for s in ("setup", "pkg") for k in ("exec_eval", "proc", "net", "decode")]
configs = {"behavior, no size columns": base, "+ module-level execution": base + top}


def model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                  random_state=0, class_weight="balanced_subsample")


def recall_at(yt, s, target, sw):
    fpr, tpr, _ = roc_curve(yt, s, sample_weight=sw)
    return tpr[fpr <= target].max()


rows = []
for name, c in configs.items():
    X = df[c].values
    auc, r1, r5 = [], [], []
    for seed in range(3):
        oof = np.zeros(len(X))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
            oof[te] = model().fit(X[tr], y[tr], sample_weight=w[tr]).predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y, oof, sample_weight=w))
        r1.append(recall_at(y, oof, 0.01, w))
        r5.append(recall_at(y, oof, 0.05, w))
    f = lambda v: f"{np.mean(v):.3f}+-{np.std(v):.3f}"
    rows.append({"features": name, "n_feat": len(c), "AUC_wtd": f(auc),
                 "recall@1%FPR": f(r1), "recall@5%FPR": f(r5)})
print(pd.DataFrame(rows).to_string(index=False))

print("\nshare of packages with module-level risky calls (any of the 4):")
anytop = (df[top].sum(axis=1) > 0)
print("  malicious: %.1f%% | benign: %.1f%%" % (100 * anytop[y == 1].mean(), 100 * anytop[y == 0].mean()))
print("compare: share with those calls anywhere (earlier rules' inputs):")
anyall = (df[[f"c_{s}_{k}" for s in ("setup", "pkg") for k in ("exec_eval", "proc", "net", "decode")]]
          .sum(axis=1) > 0)
print("  malicious: %.1f%% | benign: %.1f%%" % (100 * anyall[y == 1].mean(), 100 * anyall[y == 0].mean()))
