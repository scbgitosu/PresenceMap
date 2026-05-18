"""Parquet writers for v2 training sessions.

A ``SessionParquetWriter`` accumulates rows in memory and writes the full
parquet files on ``finalize()``. For the session sizes we care about (~hundreds
to a few thousand windows), this is simpler than a streaming ParquetWriter
and crash-safe to within the last buffered batch.

Files written under ``data/survey_projects/<project>/sessions/<id>/``:

- rf_windows.parquet
- labels.parquet
- yolo_frames.parquet
- thumbnails/index.parquet
- health.parquet

For the live ``Train`` page in the dashboard we expose ``snapshot_rf_windows()``
returning the in-memory pandas DataFrame so the UI can render without waiting
for ``finalize()``.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from shared.project import session_paths
from shared.versioning import SCHEMA_VERSION


def _flatten_window(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a WindowMsg dict into one rf_windows.parquet row."""
    rssi = msg.get("rssi", {}) or {}
    csi = msg.get("csi", {}) or {}
    return {
        "window_id": msg.get("window_id"),
        "session_id": msg.get("session_id"),
        "agent_id": msg.get("agent_id"),
        "ts_start": msg.get("ts_start"),
        "ts_end": msg.get("ts_end"),
        "phase": msg.get("phase"),
        "csi_features": list(csi.get("features", []) or []),
        "csi_frames": int(csi.get("frames", 0) or 0),
        "csi_loss_ratio": float(csi.get("loss_ratio", 1.0) or 1.0),
        "csi_amp_matrix_f16": csi.get("amp_matrix_f16"),
        "target_rssi_avg_dbm": rssi.get("target_rssi_avg_dbm"),
        "target_rssi_std_db": rssi.get("target_rssi_std_db"),
        "snr_db": rssi.get("snr_db"),
        "noise_dbm": rssi.get("noise_dbm"),
        "visible_bssid_count": rssi.get("visible_bssid_count", 0),
        "neighbor_rssi_sum_dbm": rssi.get("neighbor_rssi_sum_dbm"),
        "channel_utilization_proxy": rssi.get("channel_utilization_proxy"),
        "target_seen": bool(rssi.get("target_seen", False)),
        "interface": msg.get("interface", ""),
        "backend": msg.get("backend", ""),
        "schema_version": SCHEMA_VERSION,
    }


def _flatten_health(msg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ts": msg.get("ts"),
        "session_id": msg.get("session_id"),
        "agent_id": msg.get("agent_id"),
        "code": msg.get("code"),
        "detail": msg.get("detail", ""),
        "metrics_json": json.dumps(msg.get("metrics") or {}),
        "schema_version": SCHEMA_VERSION,
    }


class SessionParquetWriter:
    """Thread-safe in-memory buffer that materializes parquet on ``finalize()``."""

    def __init__(self, project_dir: Path | str, session_id: str) -> None:
        self.paths = session_paths(project_dir, session_id)
        self.session_id = session_id
        self.paths["session_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["thumbnails_dir"].mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._windows: List[Dict[str, Any]] = []
        self._labels: List[Dict[str, Any]] = []
        self._yolo_frames: List[Dict[str, Any]] = []
        self._healths: List[Dict[str, Any]] = []
        self._thumbnails: List[Dict[str, Any]] = []

    # ---- writers ---------------------------------------------------------- #

    def write_window(self, window_msg: Dict[str, Any]) -> None:
        row = _flatten_window(window_msg)
        with self._lock:
            self._windows.append(row)

    def write_health(self, health_msg: Dict[str, Any]) -> None:
        row = _flatten_health(health_msg)
        with self._lock:
            self._healths.append(row)

    def write_yolo_frame(self, row: Dict[str, Any]) -> None:
        row.setdefault("schema_version", SCHEMA_VERSION)
        with self._lock:
            self._yolo_frames.append(row)

    def write_label(self, row: Dict[str, Any]) -> None:
        row.setdefault("schema_version", SCHEMA_VERSION)
        with self._lock:
            self._labels.append(row)

    def write_thumbnail_index(self, row: Dict[str, Any]) -> None:
        row.setdefault("schema_version", SCHEMA_VERSION)
        with self._lock:
            self._thumbnails.append(row)

    # ---- snapshots -------------------------------------------------------- #

    def snapshot(self) -> Dict[str, pd.DataFrame]:
        with self._lock:
            return {
                "rf_windows": pd.DataFrame(self._windows),
                "labels": pd.DataFrame(self._labels),
                "yolo_frames": pd.DataFrame(self._yolo_frames),
                "health": pd.DataFrame(self._healths),
                "thumbnails": pd.DataFrame(self._thumbnails),
            }

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                "rf_windows": len(self._windows),
                "labels": len(self._labels),
                "yolo_frames": len(self._yolo_frames),
                "health": len(self._healths),
                "thumbnails": len(self._thumbnails),
            }

    # ---- finalize --------------------------------------------------------- #

    def finalize(self) -> None:
        """Write all accumulated buffers to parquet."""
        snaps = self.snapshot()
        if not snaps["rf_windows"].empty:
            _write_parquet(self.paths["rf_windows_parquet"], snaps["rf_windows"])
        if not snaps["labels"].empty:
            _write_parquet(self.paths["labels_parquet"], snaps["labels"])
        if not snaps["yolo_frames"].empty:
            _write_parquet(self.paths["yolo_frames_parquet"], snaps["yolo_frames"])
        if not snaps["health"].empty:
            _write_parquet(self.paths["health_parquet"], snaps["health"])
        if not snaps["thumbnails"].empty:
            _write_parquet(self.paths["thumbnails_index_parquet"], snaps["thumbnails"])


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to parquet, coercing list/binary columns explicitly."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def write_session_json(
    project_dir: Path | str,
    session_id: str,
    *,
    payload: Dict[str, Any],
) -> Path:
    """Materialize ``session.json`` -- created **before** any windows land."""
    p = session_paths(project_dir, session_id)
    p["session_dir"].mkdir(parents=True, exist_ok=True)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("session_id", session_id)
    p["session_json"].write_text(json.dumps(payload, indent=2, default=str))
    return p["session_json"]


def read_session_json(project_dir: Path | str, session_id: str) -> Optional[Dict[str, Any]]:
    p = session_paths(project_dir, session_id)["session_json"]
    if not p.exists():
        return None
    return json.loads(p.read_text())


def read_session_parquet(
    project_dir: Path | str, session_id: str, *, kind: str
) -> Optional[pd.DataFrame]:
    """Read one of {rf_windows, labels, yolo_frames, health, thumbnails}."""
    paths = session_paths(project_dir, session_id)
    mapping = {
        "rf_windows": paths["rf_windows_parquet"],
        "labels": paths["labels_parquet"],
        "yolo_frames": paths["yolo_frames_parquet"],
        "health": paths["health_parquet"],
        "thumbnails": paths["thumbnails_index_parquet"],
    }
    p = mapping.get(kind)
    if p is None or not p.exists():
        return None
    return pq.read_table(p).to_pandas()


def list_v2_sessions(project_dir: Path | str) -> List[str]:
    p = session_paths(project_dir, "_").get("session_dir").parent
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())
