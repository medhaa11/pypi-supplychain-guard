import tarfile
import zipfile

import duckdb
import pandas as pd

con = duckdb.connect("data/packages.duckdb")
ben = con.execute("SELECT archive_path FROM benign_random WHERE archive_path IS NOT NULL").df()

rows = []
for p in ben["archive_path"]:
    try:
        if p.endswith(".zip"):
            names = zipfile.ZipFile(p).namelist()
        else:
            names = tarfile.open(p).getnames()   # names only, nothing extracted
    except Exception as e:
        print("unreadable:", p, e)
        continue
    rows.append({
        "n_files": len(names),
        "n_py": sum(n.endswith(".py") for n in names),
        "has_git": any("/.git/" in n for n in names),
        "has_pkg_info": any(n.endswith("/PKG-INFO") for n in names),
        "has_setup_py": any(n.endswith("/setup.py") for n in names),
        "has_pyproject": any(n.endswith("/pyproject.toml") for n in names),
    })

b = pd.DataFrame(rows)
m = con.execute("""
    SELECT n_files, n_py, has_git, has_pkg_info, has_setup_py, has_pyproject
    FROM malicious_archive_info WHERE category='malicious_intent'
""").df()

cols = ["has_pkg_info", "has_setup_py", "has_pyproject", "has_git"]
print(pd.DataFrame({
    "benign_%": (b[cols].mean() * 100).round(1),
    "malicious_%": (m[cols].astype(float).mean() * 100).round(1),
}))
print("median files   benign:", int(b["n_files"].median()), "| malicious:", int(m["n_files"].median()))
print("median .py     benign:", int(b["n_py"].median()), "| malicious:", int(m["n_py"].median()))
con.close()
