import duckdb
import pandas as pd

from src.detect.rules import apply_rules

con = duckdb.connect("data/packages.duckdb")
df = con.execute("""
    SELECT x.*, s.family
    FROM features x JOIN split s USING (archive_path)
    WHERE s.split = 'train'
""").df()
con.close()

R = apply_rules(df)
for fmt in ("sdist", "wheel"):
    m = df.fmt == fmt
    mal, ben = m & (df.label == 1), m & (df.label == 0)
    print(f"\n=== {fmt} (train): {int(mal.sum())} malicious, {int(ben.sum())} benign ===")
    rows = [{"rule": c,
             "fires_on_malicious_%": round(100 * R.loc[mal, c].mean(), 1),
             "fires_on_benign_%": round(100 * R.loc[ben, c].mean(), 1)} for c in R.columns]
    print(pd.DataFrame(rows).to_string(index=False))
    fam_recall = R.loc[mal, "any"].groupby(df.loc[mal, "family"]).mean().mean()
    print(f"ANY rule: recall={100 * R.loc[mal, 'any'].mean():.1f}% | "
          f"false-positive rate={100 * R.loc[ben, 'any'].mean():.1f}% | "
          f"family-level recall={100 * fam_recall:.1f}%")
