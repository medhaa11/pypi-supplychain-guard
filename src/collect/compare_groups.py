import tarfile
import zipfile

import duckdb
import pandas as pd

con = duckdb.connect("data/packages.duckdb")


def list_names(p):
    if p.endswith((".zip", ".whl")):
        return zipfile.ZipFile(p).namelist()
    return tarfile.open(p).getnames()  # names only, nothing extracted


def summarize(paths, group):
    rows = []
    for p in paths:
        try:
            names = list_names(p)
        except Exception as e:
            print("unreadable:", p, e)
            continue
        if len(names) <= 1:  # skip empty archives
            continue
        rows.append({
            "group": group,
            "fmt": "wheel" if any(".dist-info/" in n for n in names) else "sdist",
            "n_files": len(names),
            "n_py": sum(n.endswith(".py") for n in names),
            "has_setup_py": any(n == "setup.py" or n.endswith("/setup.py") for n in names),
            "has_pyproject": any(n.endswith("/pyproject.toml") for n in names),
        })
    return pd.DataFrame(rows)


mal = con.execute(
    "SELECT archive_path FROM malicious WHERE category='malicious_intent'"
).df()["archive_path"]
ben = con.execute(
    "SELECT archive_path FROM benign_matched WHERE archive_path IS NOT NULL"
).df()["archive_path"]

df = pd.concat([summarize(mal, "malicious"), summarize(ben, "benign")])
g = df.groupby(["fmt", "group"]).agg(
    n=("n_files", "size"),
    pct_setup_py=("has_setup_py", lambda s: round(100 * s.mean(), 1)),
    pct_pyproject=("has_pyproject", lambda s: round(100 * s.mean(), 1)),
    med_files=("n_files", "median"),
    med_py=("n_py", "median"),
)
print(g)
con.close()
