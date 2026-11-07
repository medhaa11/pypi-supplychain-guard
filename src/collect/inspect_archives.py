import re
import zipfile

import duckdb
import pandas as pd

con = duckdb.connect("data/packages.duckdb")
mal = con.execute("SELECT archive_path, name, version, category FROM malicious").df()

SDIST_PKGINFO = re.compile(r"^[^/]+/([^/]+)/PKG-INFO$")
NAME_VER = re.compile(r"^(.+?)-(\d.*)$")

rows = []
for r in mal.itertuples():
    try:
        names = zipfile.ZipFile(r.archive_path).namelist()
    except Exception as e:
        rows.append({"archive_path": r.archive_path, "error": str(e)})
        continue
    recovered = None
    for n in names:
        m = SDIST_PKGINFO.match(n)
        if m:
            nv = NAME_VER.match(m.group(1))
            if nv:
                recovered = nv.group(2)
            break
    rows.append({
        "archive_path": r.archive_path,
        "category": r.category,
        "n_files": len(names),
        "n_py": sum(n.endswith(".py") for n in names),
        "has_git": any("/.git/" in n for n in names),
        "has_pkg_info": any(n.endswith("/PKG-INFO") for n in names),
        "has_setup_py": any(n.endswith("/setup.py") for n in names),
        "has_pyproject": any(n.endswith("/pyproject.toml") for n in names),
        "has_whl": any(n.endswith(".whl") for n in names),
        "has_targz": any(n.endswith(".tar.gz") for n in names),
        "recovered_version": recovered,
        "error": None,
    })

df = pd.DataFrame(rows)
print("errors:", int(df["error"].notna().sum()))
ok = df[df["error"].isna()]
print(ok.groupby("category")[["has_git", "has_pkg_info", "has_setup_py",
                              "has_pyproject", "has_whl", "has_targz"]].sum())
print("median files:", int(ok["n_files"].median()), "| median .py files:", int(ok["n_py"].median()))
unv = mal[mal["version"].isna()]["archive_path"]
print("unversioned:", len(unv),
      "| version recovered:", int(ok[ok["archive_path"].isin(unv)]["recovered_version"].notna().sum()))
con.execute("CREATE OR REPLACE TABLE malicious_archive_info AS SELECT * FROM df")
con.close()
