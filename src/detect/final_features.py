SCOPES = ("setup", "pkg")
SIZEISH = {"lines", "bytes", "files", "calls"}


def feature_columns(columns):
    base = [c for c in columns if c.startswith("c_") and "_top_" not in c
            and c.split("_", 2)[2] not in SIZEISH]
    top = [f"c_{s}_top_{k}" for s in SCOPES for k in ("exec_eval", "proc", "net", "decode")]
    return base + top
