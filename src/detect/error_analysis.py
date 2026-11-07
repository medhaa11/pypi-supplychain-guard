import duckdb
import pandas as pd

from src.detect.rules import apply_rules

con = duckdb.connect("data/packages.duckdb")
df = con.execute("""
    SELECT x.*, s.family, f.name_stem
    FROM features x
    JOIN split s USING (archive_path)
    JOIN families f USING (archive_path)
    WHERE s.split = 'train' AND x.fmt = 'sdist'
""").df()
con.close()

R = apply_rules(df)
df["hit"] = R["any"].values
df["code_bytes"] = df.c_setup_bytes + df.c_pkg_bytes
df["any_risky_call"] = (df[["c_setup_net", "c_pkg_net", "c_setup_proc", "c_pkg_proc",
                            "c_setup_exec_eval", "c_pkg_exec_eval",
                            "c_setup_decode", "c_pkg_decode"]].sum(axis=1) > 0)
df["obf_signal"] = (df[["c_setup_obf", "c_pkg_obf", "c_setup_b64_blob", "c_pkg_b64_blob",
                        "c_setup_hi_ent_str", "c_pkg_hi_ent_str"]].sum(axis=1) > 0)

mal = df[df.label == 1]
caught, missed = mal[mal.hit], mal[~mal.hit]
print(f"malicious sdists: caught={len(caught)}, missed={len(missed)}")
for name, g in (("caught", caught), ("missed", missed)):
    print(f"\n-- {name} --")
    print("median code bytes:", int(g.code_bytes.median()),
          "| has setup.py: %.0f%%" % (100 * g.s_has_setup_py.mean()),
          "| any net/proc/exec/decode call: %.0f%%" % (100 * g.any_risky_call.mean()),
          "| obfuscation-like signal: %.0f%%" % (100 * g.obf_signal.mean()),
          "| has URL string: %.0f%%" % (100 * ((g.c_setup_urls + g.c_pkg_urls) > 0).mean()))
print("\nmissed malicious: share with NO risky call at all: %.0f%%" %
      (100 * (~missed.any_risky_call).mean()))
print("missed malicious: share under 500 bytes of code: %.0f%%" %
      (100 * (missed.code_bytes < 500).mean()))
print("\nlargest name stems among MISSED malicious:")
print(missed.name_stem.value_counts().head(8).to_string())

ben = df[df.label == 0]
fp = ben[ben.hit].copy()
print(f"\nbenign sdists flagged: {len(fp)} of {len(ben)}")
rules_only = R.drop(columns="any")
fp_rules = rules_only.loc[fp.index].apply(lambda r: ", ".join(c[:2] for c, v in r.items() if v), axis=1)
print("flagged benign (name, which rules):")
for n, r in list(zip(fp.name, fp_rules))[:20]:
    print("  ", n[:40], "->", r)
