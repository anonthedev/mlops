"""Settings and small file helpers shared by every module.

All paths and thresholds live in params.yaml at the repository root so that
nothing is hard-coded in the code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

# Repository root = two folders above this file (src/mp124/config.py).
ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    """Read params.yaml with yaml.safe_load (never yaml.load: it can run code)."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str) -> Path:
    """Turn a path from params.yaml (relative to the repo root) into an absolute path."""
    return ROOT / rel_path


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the raw bytes of a file. The marking tool uses this as the
    fingerprint of our train.csv, so we hash the original file, never a copy
    that pandas or Excel has rewritten."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(name: str, params: dict | None = None) -> pd.DataFrame:
    """Load one of the data files named in params.yaml (train, later or holdout)."""
    params = params or load_params()
    return pd.read_csv(resolve(params["data"][name]))
