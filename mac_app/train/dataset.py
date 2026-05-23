"""Parquet I/O for ESP32 training sessions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from shared.project import session_paths
from shared.versioning import SCHEMA_VERSION


class SessionParquetWriter:
    """In-memory buffer; writes rf_windows + labels on finalize."""

    def __init__(self, project_dir: Path | str, session_id: str) -> None:
        self.paths = session_paths(project_dir, session_id)
        self.session_id = session_id
        self.paths["session_dir"].mkdir(parents=True, exist_ok=True)
        self._windows: List[Dict[str, Any]] = []
        self._labels: List[Dict[str, Any]] = []

    def write_rf_window(self, row: Dict[str, Any]) -> None:
        row.setdefault("schema_version", SCHEMA_VERSION)
        row.setdefault("session_id", self.session_id)
        row.setdefault("backend", "esp32")
        self._windows.append(row)

    def write_label(self, row: Dict[str, Any]) -> None:
        row.setdefault("schema_version", SCHEMA_VERSION)
        self._labels.append(row)

    def finalize(self) -> None:
        if self._windows:
            _write_parquet(self.paths["rf_windows_parquet"], pd.DataFrame(self._windows))
        if self._labels:
            _write_parquet(self.paths["labels_parquet"], pd.DataFrame(self._labels))


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def write_session_json(
    project_dir: Path | str,
    session_id: str,
    *,
    payload: Dict[str, Any],
) -> Path:
    p = session_paths(project_dir, session_id)
    p["session_dir"].mkdir(parents=True, exist_ok=True)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("session_id", session_id)
    payload.setdefault("backend", "esp32")
    p["session_json"].write_text(json.dumps(payload, indent=2, default=str))
    return p["session_json"]


def read_session_parquet(
    project_dir: Path | str, session_id: str, *, kind: str
) -> Optional[pd.DataFrame]:
    paths = session_paths(project_dir, session_id)
    mapping = {
        "rf_windows": paths["rf_windows_parquet"],
        "labels": paths["labels_parquet"],
    }
    p = mapping.get(kind)
    if p is None or not p.exists():
        return None
    return pq.read_table(p).to_pandas()


def list_sessions(project_dir: Path | str) -> List[str]:
    p = session_paths(project_dir, "_")["session_dir"].parent
    if not p.exists():
        return []
    return sorted(
        d.name
        for d in p.iterdir()
        if d.is_dir() and (d / "rf_windows.parquet").exists()
    )


# Back-compat alias
list_v2_sessions = list_sessions
