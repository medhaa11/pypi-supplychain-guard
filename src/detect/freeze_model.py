import hashlib
import json
import subprocess
import warnings

import duckdb
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src.detect.final_features import feature_columns

warnings.filterwarnings("ignore")

con = duckdb.connect("data/packages.duckdb")
df = con.execute("""
    SELECT x.*, s.family AS cluster
    FROM features x JOIN split2 s USING (archive_path)
    WHERE s.split = 'train' AND x.fmt = 'sdist'
""").df().reset_index(drop=True)
con.close()

cols = feature_columns(df.columns)
X, y, groups = df[cols].values, df.label.values, df.cluster.values
w = 1.0 / df.groupby("cluster")["cluster"].transform("size").values
print(f"train sdists: {len(df)} rows | clusters: {df.cluster.nunique()} | features: {len(cols)}")


def model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                  random_state=0, class_weight="balanced_subsample")


oof = np.zeros(len(df))
for seed in range(3):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        oof[te] += model().fit(X[tr], y[tr], sample_weight=w[tr]).predict_proba(X[te])[:, 1] / 3


def threshold_for(fpr):
    b, wb = oof[y == 0], w[y == 0] / w[y == 0].sum()
    order = np.argsort(-b)
    k = int(np.searchsorted(np.cumsum(wb[order]), fpr, side="right"))
    return float(b[order][k]) if k < len(b) else -1.0  # flag if score > threshold


out = {}
for name, fpr in (("1pct", 0.01), ("5pct", 0.05)):
    t = threshold_for(fpr)
    flag = oof > t
    rec = np.average(flag[y == 1], weights=w[y == 1])
    fp = np.average(flag[y == 0], weights=w[y == 0])
    print(f"threshold for {name} FPR: {t:.4f} -> train-CV cluster-weighted recall={rec:.3f}, FPR={fp:.3f}")
    out[f"threshold_{name}"] = t

final = model().fit(X, y, sample_weight=w)
joblib.dump(final, "data/final_model.joblib")
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
out.update(features=cols, n_train_rows=len(df), n_train_clusters=int(df.cluster.nunique()),
           model="RandomForest(300 trees, min_samples_leaf=2, balanced_subsample, seed 0)",
           cluster_weighted=True, git_commit_before_freeze=commit,
           model_sha256=hashlib.sha256(open("data/final_model.joblib", "rb").read()).hexdigest())
json.dump(out, open("results/frozen_model.json", "w"), indent=2)
print("\nsaved data/final_model.joblib and results/frozen_model.json")
