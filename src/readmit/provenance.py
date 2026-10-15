"""Data provenance: hashes, row counts, library versions and stage lineage."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

TRACKED_LIBS = ["numpy", "pandas", "scikit-learn", "lightgbm", "shap", "streamlit", "pyarrow", "plotly"]


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def library_versions() -> dict[str, str]:
    out = {}
    for lib in TRACKED_LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "not-installed"
    out["python"] = platform.python_version()
    return out


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class Provenance:
    """Append-only lineage log written to artifacts/provenance.json."""

    def __init__(self, path: Path, config_fingerprint: str, root: Path | None = None):
        self.path = Path(path)
        self.fingerprint = config_fingerprint
        self.root = root
        self.doc = {"stages": []}
        if self.path.exists():
            try:
                self.doc = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.doc = {"stages": []}
        self.doc.setdefault("stages", [])

    def record(self, stage: str, inputs: list[Path], outputs: list[Path], notes: dict | None = None,
               row_counts: dict[str, int] | None = None) -> None:
        entry = {
            "stage": stage,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_fingerprint": self.fingerprint,
            "inputs": [{"path": str(p), "sha256": file_sha256(Path(p))} for p in inputs if Path(p).exists()],
            "outputs": [{"path": str(p), "sha256": file_sha256(Path(p)), "bytes": Path(p).stat().st_size} for p in outputs if Path(p).exists()],
            "row_counts": row_counts or {},
            "notes": notes or {},
        }
        # replace an earlier entry for the same stage (re-runs overwrite, not accumulate)
        self.doc["stages"] = [s for s in self.doc["stages"] if s["stage"] != stage] + [entry]
        self.doc["environment"] = {"libraries": library_versions(), "git_commit": git_commit(self.root) if self.root else None}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.doc, indent=2, default=str))


def table_row_counts(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {k: int(len(v)) for k, v in tables.items()}
