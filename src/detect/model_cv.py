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
    SELECT x.*, s.family
    FROM features x JOIN split s USING (archive_path)
    WHERE s.split = 'train'
""").df()
con.close()

SCOPES = ("setup", "pkg")
BEH = ["calls", "exec_eval", "dyn_import", "proc", "net", "decode", "env", "sysinfo", "fs", "obf",
       "imp_net", "imp_proc", "cmdclass", "cmd_subclass", "long_str", "hi_ent_str", "b64_blob",
       "sens", "urls", "ips", "exfil", "parse_fail", "max_ent"]
SIZE = ["files", "lines", "bytes"]
beh = [f"c_{s}_{k}" for s in SCOPES for k in BEH]
size = [f"c_{s}_{k}" for s in SCOPES for k in SIZE]
struct = ["s_n_files", "s_n_py", "s_has_setup_py", "s_has_pyproject", "s_has_setup_cfg"]
meta = [c for c in df.columns if c.startswith("m_")]
configs = {
    "A behavior only": beh,
    "B + size": beh + size,
    "C + structural + metadata": beh + size + struct + meta,
}

y = df.label.values
groups = df.family.values
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)


def recall_at_fpr(yt, s, target):
    fpr, tpr, _ = roc_curve(yt, s)
    return tpr[fpr <= target].max()


def make_model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                  random_state=0, class_weight="balanced_subsample")


rows = []
for name, cols in configs.items():
    X = df[cols].values
    oof = np.zeros(len(df))
    for tr, te in cv.split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    for fmt in ("sdist", "wheel"):
        mk = (df.fmt == fmt).values
        rows.append({
            "features": name, "fmt": fmt, "n": int(mk.sum()),
            "AUC": round(roc_auc_score(y[mk], oof[mk]), 3),
            "recall@1%FPR": round(recall_at_fpr(y[mk], oof[mk], 0.01), 3),
            "recall@5%FPR": round(recall_at_fpr(y[mk], oof[mk], 0.05), 3),
        })
print(pd.DataFrame(rows).to_string(index=False))

R = apply_rules(df)
print("\nrule baseline (same train rows):")
for fmt in ("sdist", "wheel"):
    mk = (df.fmt == fmt).values
    hit = R["any"].values
    print(f"  {fmt}: recall={hit[mk & (y == 1)].mean():.3f} at FPR={hit[mk & (y == 0)].mean():.3f}")

X = df[configs["A behavior only"]].values
m = make_model().fit(X, y)
imp = pd.Series(m.feature_importances_, index=configs["A behavior only"]).sort_values(ascending=False)
print("\ntop features (config A, impurity importance, rough guide only):")
print(imp.head(10).round(3).to_string())
