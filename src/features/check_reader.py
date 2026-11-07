import re
import time
from collections import Counter

import duckdb
import pandas as pd

from src.ecosystems.pypi_reader import read_archive

con = duckdb.connect("data/packages.duckdb")
groups = {
    "malicious": con.execute(
        "SELECT archive_path FROM malicious WHERE category='malicious_intent'"
    ).df()["archive_path"].tolist(),
    "benign": con.execute(
        "SELECT archive_path FROM benign_matched WHERE archive_path IS NOT NULL"
    ).df()["archive_path"].tolist(),
}
DATE_LEFT = re.compile(r"^\d{4}-\d{2}-\d{2}-")
t0, rows = time.time(), []

for g, paths in groups.items():
    errors, empty, no_py = 0, 0, 0
    leftover, nested, offenders = 0, 0, []
    for p in paths:
        try:
            a = read_archive(p)
        except Exception:
            errors += 1
            continue
        if not a.all_paths:
            empty += 1
            continue
        s = set(a.all_paths)
        no_py += not any(x.endswith(".py") for x in a.all_paths)
        bad = [x for x in a.all_paths if DATE_LEFT.match(x) or x.startswith("tmp/")]
        if bad:
            leftover += 1
            offenders.append(bad[0])
        # project root still one folder down? (PKG-INFO/setup.py exists, but not at top)
        if not (s & {"PKG-INFO", "setup.py", "pyproject.toml"}) and any(
                x.count("/") == 1 and x.split("/")[1] in {"PKG-INFO", "setup.py", "pyproject.toml"}
                for x in a.all_paths):
            nested += 1
            offenders.append(next(x for x in a.all_paths if x.count("/") == 1))
        fmt = "wheel" if any(".dist-info/" in x for x in a.all_paths) else "sdist"
        rows.append({
            "group": g, "fmt": fmt, "n_files": len(a.all_paths),
            "n_py": sum(x.endswith(".py") for x in a.all_paths),
            "has_setup_py": "setup.py" in s, "has_pyproject": "pyproject.toml" in s,
        })
    print(f"\n=== {g}: {len(paths)} archives ===")
    print("errors:", errors, "| empty:", empty, "| no .py:", no_py)
    print("leftover date/tmp wrapper:", leftover, "| project root still nested:", nested)
    if offenders:
        print("example offending paths:", offenders[:5])

df = pd.DataFrame(rows)
print("\n=== structure comparison ===")
print(df.groupby(["fmt", "group"]).agg(
    n=("n_files", "size"),
    pct_setup_py=("has_setup_py", lambda s: round(100 * s.mean(), 1)),
    pct_pyproject=("has_pyproject", lambda s: round(100 * s.mean(), 1)),
    med_files=("n_files", "median"),
    med_py=("n_py", "median"),
))
print(f"\nelapsed: {time.time() - t0:.0f}s")
con.close()
