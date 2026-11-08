"""Static feature extraction. Parses code with ast (never executes or imports it)."""
import ast
import math
import re
import time
import warnings
from collections import Counter
from email.parser import Parser

import duckdb
import pandas as pd

from src.ecosystems.pypi_reader import read_archive

warnings.filterwarnings("ignore")

SYSINFO = {"getpass.getuser", "socket.gethostname", "socket.gethostbyname", "platform.node",
           "platform.system", "platform.uname", "platform.platform", "os.getlogin", "uuid.getnode"}
NET_PREFIX = ("requests.", "urllib.request.", "urllib.urlopen", "urllib2.", "urllib3.",
              "http.client.", "httpx.", "aiohttp.", "socket.", "ftplib.", "smtplib.",
              "telnetlib.", "websocket.", "paramiko.", "webbrowser.")
NET_ROOTS = {"requests", "urllib", "urllib2", "urllib3", "http", "httpx", "aiohttp", "socket",
             "ftplib", "smtplib", "telnetlib", "websocket", "paramiko"}
PROC_ROOTS = {"subprocess", "pty"}
DECODE = {"base64.b64decode", "base64.b32decode", "base64.b85decode", "base64.a85decode",
          "base64.decodebytes", "binascii.unhexlify", "binascii.a2b_base64", "zlib.decompress",
          "marshal.loads", "codecs.decode", "pickle.loads", "lzma.decompress", "bz2.decompress",
          "gzip.decompress"}
EXEC = {"eval", "exec", "compile"}
DYN = {"__import__", "importlib.import_module"}
FS = {"os.walk", "os.listdir", "glob.glob", "os.path.expanduser", "pathlib.Path.home", "os.scandir"}
OBF = {"chr", "getattr", "globals", "locals", "setattr"}
CMD_BASES = {"install", "develop", "egg_info", "build_py", "build_ext", "sdist", "bdist_wheel", "Command"}

RE_SENS = re.compile(rb"\.ssh|id_rsa|\.aws|/etc/passwd|/etc/shadow|\.npmrc|\.pypirc|\.git-credentials|"
                     rb"\.docker/config|Login Data|Local State|wallet|keychain|\.kube/config", re.I)
RE_URL = re.compile(rb"https?://[^\s'\"<>]+")
RE_IP = re.compile(rb"\b\d{1,3}(?:\.\d{1,3}){3}\b")
RE_EXFIL = re.compile(rb"discord(?:app)?\.com/api/webhooks|pastebin|ngrok|telegram|transfer\.sh|"
                      rb"webhook\.site|requestbin|pipedream|interact\.sh|oast\.|burpcollaborator|"
                      rb"anonfiles|paste\.ee|hastebin", re.I)
RE_B64 = re.compile(r"^[A-Za-z0-9+/=\s]+$")

KEYS = ["files", "lines", "bytes", "parse_fail", "calls", "exec_eval", "dyn_import", "proc", "net",
        "decode", "env", "sysinfo", "fs", "obf", "imp_net", "imp_proc", "cmdclass", "cmd_subclass",
        "long_str", "hi_ent_str", "b64_blob", "sens", "urls", "ips", "exfil",
        "top_calls", "top_exec_eval", "top_proc", "top_net", "top_decode"]


def entropy(s):
    c, n = Counter(s), len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def analyze_py(src, is_setup=False):
    f = Counter()
    f["lines"] = src.count(b"\n") + 1
    f["bytes"] = len(src)
    f["sens"] = len(RE_SENS.findall(src))
    f["urls"] = len(RE_URL.findall(src))
    f["ips"] = len(RE_IP.findall(src))
    f["exfil"] = len(RE_EXFIL.findall(src))
    try:
        tree = ast.parse(src)
    except Exception:
        f["parse_fail"] = 1
        return f

    nodes = list(ast.walk(tree))
    alias = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    alias[a.asname] = a.name
                else:
                    alias[a.name.split(".")[0]] = a.name.split(".")[0]
                root = a.name.split(".")[0]
                f["imp_net"] += root in NET_ROOTS
                f["imp_proc"] += root in PROC_ROOTS
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                alias[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
            root = mod.split(".")[0]
            f["imp_net"] += root in NET_ROOTS
            f["imp_proc"] += root in PROC_ROOTS

    def resolve(n):
        first, *rest = n.split(".")
        return ".".join([alias.get(first, first)] + rest)

    def top_level(n):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if (not is_setup and isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__"):
            return
        if isinstance(n, ast.Call):
            d = dotted(n.func)
            if d:
                d = resolve(d)
                f["top_calls"] += 1
                if d in EXEC:
                    f["top_exec_eval"] += 1
                if (d.startswith("subprocess.") or d in ("os.system", "os.popen", "pty.spawn")
                        or d.startswith(("os.spawn", "os.exec"))):
                    f["top_proc"] += 1
                if d.startswith(NET_PREFIX) and d not in SYSINFO and not d.startswith("urllib.parse."):
                    f["top_net"] += 1
                if d in DECODE:
                    f["top_decode"] += 1
        for c in ast.iter_child_nodes(n):
            top_level(c)

    try:
        top_level(tree)
    except RecursionError:
        pass

    max_ent = 0.0
    for node in nodes:
        if isinstance(node, ast.Call):
            n = dotted(node.func)
            if n:
                n = resolve(n)
                f["calls"] += 1
                if n in EXEC:
                    f["exec_eval"] += 1
                if n in DYN:
                    f["dyn_import"] += 1
                if (n.startswith("subprocess.") or n in ("os.system", "os.popen", "pty.spawn")
                        or n.startswith(("os.spawn", "os.exec"))):
                    f["proc"] += 1
                if n in SYSINFO:
                    f["sysinfo"] += 1
                elif n.startswith(NET_PREFIX) and not n.startswith("urllib.parse."):
                    f["net"] += 1
                if n in DECODE:
                    f["decode"] += 1
                if n == "os.getenv":
                    f["env"] += 1
                if n in FS:
                    f["fs"] += 1
                if n in OBF:
                    f["obf"] += 1
            for kw in node.keywords:
                if kw.arg == "cmdclass":
                    f["cmdclass"] += 1
        elif isinstance(node, ast.Attribute):
            n = dotted(node)
            if n and resolve(n) == "os.environ":
                f["env"] += 1
        elif isinstance(node, ast.ClassDef):
            for b in node.bases:
                bn = dotted(b)
                if bn and bn.rsplit(".", 1)[-1] in CMD_BASES:
                    f["cmd_subclass"] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= 40:
            s = node.value
            f["long_str"] += 1
            e = entropy(s)
            max_ent = max(max_ent, e)
            if e >= 4.5:
                f["hi_ent_str"] += 1
            if len(s) >= 100 and RE_B64.match(s):
                f["b64_blob"] += 1
    f["max_ent"] = max_ent
    return f


def metadata_features(arc):
    text = None
    if "PKG-INFO" in arc.files:
        text = arc.files["PKG-INFO"]
    else:
        for p, d in arc.files.items():
            if p.endswith(".dist-info/METADATA"):
                text = d
                break
    out = dict(m_present=0, m_has_author=0, m_has_home=0, m_summary_len=0, m_desc_len=0,
               m_n_requires=0, m_n_classifiers=0, m_has_license=0)
    if text is None:
        return out
    try:
        msg = Parser().parsestr(text.decode("utf-8", "replace"))
    except Exception:
        return out
    body = msg.get_payload() or ""
    out.update(
        m_present=1,
        m_has_author=int(bool((msg.get("Author") or "").strip() or (msg.get("Author-email") or "").strip())),
        m_has_home=int(bool((msg.get("Home-page") or "").strip() or msg.get_all("Project-URL"))),
        m_summary_len=len(msg.get("Summary") or ""),
        m_desc_len=len(body) if isinstance(body, str) else 0,
        m_n_requires=len(msg.get_all("Requires-Dist") or []),
        m_n_classifiers=len(msg.get_all("Classifier") or []),
        m_has_license=int(bool((msg.get("License") or "").strip() not in ("", "UNKNOWN"))),
    )
    return out


def extract(path, data=None):
    arc = read_archive(path, data)
    if not arc.all_paths:
        return None
    s = set(arc.all_paths)
    row = {
        "s_n_files": len(arc.all_paths),
        "s_n_py": sum(x.endswith(".py") for x in arc.all_paths),
        "s_has_setup_py": int("setup.py" in s),
        "s_has_pyproject": int("pyproject.toml" in s),
        "s_has_setup_cfg": int("setup.cfg" in s),
        "s_is_wheel": int(any(".dist-info/" in x for x in arc.all_paths)),
        "s_truncated": int(arc.truncated),
        "x_mtime_max": arc.mtime_max,
    }
    scopes = {"setup": Counter(), "pkg": Counter()}
    mx = {"setup": 0.0, "pkg": 0.0}
    for p, data in arc.files.items():
        if not p.endswith(".py"):
            continue
        sc = "setup" if p == "setup.py" else "pkg"
        f = analyze_py(data, is_setup=(sc == "setup"))
        mx[sc] = max(mx[sc], f.pop("max_ent", 0.0))
        scopes[sc].update(f)
        scopes[sc]["files"] += 1
    for sc in scopes:
        for k in KEYS:
            row[f"c_{sc}_{k}"] = scopes[sc].get(k, 0)
        row[f"c_{sc}_max_ent"] = round(mx[sc], 3)
    row.update(metadata_features(arc))
    return row


if __name__ == "__main__":
    con = duckdb.connect("data/packages.duckdb")
    mal = con.execute("""SELECT archive_path, name, version, date_added AS date, category
                         FROM malicious WHERE category='malicious_intent'""").df()
    mal["group"], mal["label"] = "malicious", 1
    ben = con.execute("""SELECT archive_path, name, version, upload_time AS date
                         FROM benign_matched WHERE archive_path IS NOT NULL""").df()
    ben["group"], ben["label"], ben["category"] = "benign", 0, "benign"
    meta = pd.concat([mal, ben], ignore_index=True)

    rows, skipped, t0 = [], 0, time.time()
    for i, r in enumerate(meta.itertuples(), 1):
        try:
            feats = extract(r.archive_path)
        except Exception as e:
            print("failed:", r.archive_path.split("/")[-1], type(e).__name__)
            feats = None
        if feats is None:
            skipped += 1
            continue
        rows.append({"archive_path": r.archive_path, "name": r.name, "version": r.version,
                     "date": r.date, "group": r.group, "label": r.label,
                     "category": r.category, **feats})
        if i % 500 == 0:
            print(f"{i}/{len(meta)} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df["fmt"] = df["s_is_wheel"].map({1: "wheel", 0: "sdist"})
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM df")
    df.to_parquet("data/features.parquet")
    print("rows:", len(df), "| skipped (empty/failed):", skipped)

    # descriptive first look, NOT a result: % of packages with each behavior, by group and format
    d = pd.DataFrame({
        "parse_fail": (df.c_setup_parse_fail + df.c_pkg_parse_fail) > 0,
        "exec_eval": (df.c_setup_exec_eval + df.c_pkg_exec_eval) > 0,
        "proc": (df.c_setup_proc + df.c_pkg_proc) > 0,
        "net": (df.c_setup_net + df.c_pkg_net) > 0,
        "decode": (df.c_setup_decode + df.c_pkg_decode) > 0,
        "env": (df.c_setup_env + df.c_pkg_env) > 0,
        "sens_path": (df.c_setup_sens + df.c_pkg_sens) > 0,
        "exfil_str": (df.c_setup_exfil + df.c_pkg_exfil) > 0,
        "setup_has_net": df.c_setup_net > 0,
        "cmdclass": (df.c_setup_cmdclass + df.c_setup_cmd_subclass) > 0,
        "group": df["group"], "fmt": df["fmt"],
    })
    print((d.groupby(["fmt", "group"]).mean(numeric_only=True) * 100).round(1))
    con.close()
