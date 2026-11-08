"""Safe, in-memory reader for PyPI package archives (sdist .tar.gz/.zip, wheels).

Never writes to disk, never imports or executes anything from the archive.
Normalizes away dataset-packaging artifacts so both groups look the same.
"""
import io
import re
import time
import tarfile
import zipfile
from dataclasses import dataclass, field

MAX_FILE_BYTES = 1_000_000
MAX_FILES = 3000
MAX_TOTAL_BYTES = 30_000_000
ZIP_PWD = b"infected"  # Datadog dataset password; ignored for unencrypted zips

DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}-")
TMP_DIR = re.compile(r"^tmp[A-Za-z0-9_]*$")
SDIST_ROOT = re.compile(r".+-\d")
PKG_INFO_JSON = re.compile(r"^package_info-.*\.json$")  # Datadog-only metadata file
ROOT_MARKERS = {"PKG-INFO", "setup.py", "pyproject.toml", "setup.cfg"}
WANTED_NAMES = {"PKG-INFO", "METADATA", "entry_points.txt", "requirements.txt"}
WANTED_EXT = (".py", ".cfg", ".toml")


@dataclass
class Archive:
    all_paths: list = field(default_factory=list)
    files: dict = field(default_factory=dict)  # cleaned path -> bytes (wanted files only)
    truncated: bool = False
    mtime_max: float = 0.0


def _strip_wrappers(parts):
    for _ in range(4):
        if not parts or any(len(p) < 2 for p in parts):
            break
        roots = {p[0] for p in parts}
        if len(roots) != 1:
            break
        r = next(iter(roots))
        if (DATE_PREFIX.match(r) or TMP_DIR.match(r)
                or (SDIST_ROOT.match(r) and not r.endswith((".dist-info", ".egg-info")))):
            parts = [p[1:] for p in parts]
        else:
            break
    return parts


def _undouble(p):
    """'a/b.py/a/b.py' -> 'a/b.py' (some dataset wheels store every file twice)."""
    n = len(p)
    if n >= 2 and n % 2 == 0 and p[: n // 2] == p[n // 2:]:
        if n >= 4 or "." in p[0]:
            return p[: n // 2]
    return p


def _find_sdist_root(parts):
    """Prefix of the real project root: parent of the shallowest root-marker file."""
    best = None
    for p in parts:
        if p and p[-1] in ROOT_MARKERS and not any(c.endswith(".egg-info") for c in p[:-1]):
            if best is None or len(p) < len(best):
                best = p
    return tuple(best[:-1]) if best else None


def _clean(raw_names):
    parts = [n.strip("/").split("/") for n in raw_names]
    parts = _strip_wrappers(parts)
    parts = [_undouble(p) for p in parts]
    is_wheel = any(c.endswith('.dist-info') for q in parts for c in q[:-1])
    root = None if is_wheel else _find_sdist_root(parts)
    if root:
        k = len(root)
        parts = [p[k:] if tuple(p[:k]) == root else None for p in parts]
    out = []
    for p in parts:
        if not p:
            out.append("")  # outside the project root: drop
        elif (len(p) == 1 and PKG_INFO_JSON.match(p[0])) or p[0] == ".git":
            out.append("")  # dataset artifact: drop
        else:
            out.append("/".join(p))
    return out


def _wanted(path):
    return path.endswith(WANTED_EXT) or path.rsplit("/", 1)[-1] in WANTED_NAMES


def read_archive(path, data=None):
    """Format is decided by the suffix of `path`; `data` (bytes) replaces reading from disk."""
    entries = []  # (raw_name, size, reader_fn)
    mt = []
    is_zip = path.endswith((".zip", ".whl")) or (data is not None and data[:4] == b"PK\x03\x04")
    if is_zip:
        zf = zipfile.ZipFile(io.BytesIO(data) if data is not None else path)
        for info in zf.infolist():
            if not info.is_dir():
                entries.append((info.filename, info.file_size,
                                lambda i=info: zf.read(i, pwd=ZIP_PWD)))
                try:
                    mt.append(time.mktime(tuple(info.date_time) + (0, 0, -1)))
                except Exception:
                    pass
    else:
        tf = tarfile.open(fileobj=io.BytesIO(data)) if data is not None else tarfile.open(path)
        for m in tf.getmembers():
            if m.isfile():
                entries.append((m.name, m.size, lambda m=m: tf.extractfile(m).read()))
                mt.append(m.mtime)

    arc = Archive()
    arc.mtime_max = max(mt) if mt else 0.0
    if len(entries) > MAX_FILES:
        entries = entries[:MAX_FILES]
        arc.truncated = True
    cleaned = _clean([e[0] for e in entries])
    arc.all_paths = [c for c in cleaned if c]

    total = 0
    for (raw, size, reader), cp in zip(entries, cleaned):
        if not cp or not _wanted(cp) or size > MAX_FILE_BYTES:
            continue
        if total + size > MAX_TOTAL_BYTES:
            arc.truncated = True
            break
        arc.files[cp] = reader()
        total += size
    return arc
