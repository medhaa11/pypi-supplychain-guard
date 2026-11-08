import collections
import io
import tarfile
import zipfile

ZIP = "data/raw/malregistry_dl/main.zip"
ROOT = "pypi_malregistry-main/"
zf = zipfile.ZipFile(ZIP)
kinds = collections.Counter()
examples = {}
for info in zf.infolist():
    if info.is_dir() or not info.filename.endswith(".tar.gz") or info.file_size > 50_000_000:
        continue
    data = zf.read(info)
    try:
        tarfile.open(fileobj=io.BytesIO(data)).getmembers()
    except Exception as e:
        magic = data[:4].hex() if data else "EMPTY"
        size_bucket = "0B" if not data else "<1KB" if len(data) < 1024 else "<100KB" if len(data) < 100_000 else ">=100KB"
        key = (type(e).__name__, str(e)[:45], magic, size_bucket)
        kinds[key] += 1
        examples.setdefault(key, info.filename[len(ROOT):])
print("failing archives:", sum(kinds.values()))
for k, n in kinds.most_common(12):
    print(n, k, "e.g.", examples[k])
