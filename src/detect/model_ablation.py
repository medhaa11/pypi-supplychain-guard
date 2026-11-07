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
df["code_bytes"] = df.c_setup_bytes + df.c_pkg_bytes

print("=== largest clusters in train (sdists) ===")
for cl, g in sorted(df.groupby("cluster"), key=lambda t: -len(t[1]))[:8]:
    stems = g.name.str.lower().str.replace(r"[\d_.-]+$", "", regex=True).value_counts().head(3).to_dict()
    print(f"size={len(g):3d} malicious={int(g.label.sum()):3d} "
          f"median_bytes={int(g.code_bytes.median()):6d} dates={g.date.min()}..{g.date.max()} names={stems}")


def cols(keys):
    return [f"c_{s}_{k}" for s in ("setup", "pkg") for k in keys]


SIZEISH = {"lines", "bytes", "files", "calls"}
ALL = [c for c in df.columns if c.startswith("c_") and c.split("_", 2)[2] not in SIZEISH]
guard_strings = cols(["exfil", "urls", "ips", "sens"])
guard_hooks = cols(["cmdclass", "cmd_subclass"])
configs = {
    "size only (reference)": cols(["lines", "bytes", "files"]),
    "behavior, no size columns": ALL,
    "minus string signals (exfil/urls/ips/sens)": [c for c in ALL if c not in guard_strings],
    "also minus install-hook signals": [c for c in ALL if c not in guard_strings + guard_hooks],
    "generic API calls only": cols(["exec_eval", "dyn_import", "proc", "net", "decode", "env",
                                    "sysinfo", "fs"]),
}


def model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                  random_state=0, class_weight="balanced_subsample")


def recall_at(yt, s, target, sw):
    fpr, tpr, _ = roc_curve(yt, s, sample_weight=sw)
    return tpr[fpr <= target].max()


print("\n=== ablation: cluster-weighted, mean +- sd over 3 CV seeds ===")
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
