"""Record ESP32 windowed CSI into v2 session parquet (CSI-only rows)."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd

from mac_app.capture.esp32_ingest import Esp32IngestService, Esp32WindowRow
from shared.project import session_paths
from shared.utils import now_iso


class Esp32SessionRecorder:
    """Buffers ``Esp32WindowRow`` from ingest and writes ``rf_windows.parquet``."""

    def __init__(self, project_dir: Path, session_id: str, *, phase: str = "calibration") -> None:
        self.project_dir = Path(project_dir)
        self.session_id = session_id
        self.phase = phase
        self._rows: List[dict] = []
        self._paths = session_paths(self.project_dir, session_id)

    def on_window(self, row: Esp32WindowRow) -> None:
        self._rows.append(
            {
                "window_id": str(uuid.uuid4()),
                "ts_start": row.ts_start,
                "ts_end": row.ts_end,
                "phase": self.phase,
                "node_id": row.node_id,
                "source_ip": row.source_ip,
                "csi_features": row.features,
                "csi_frames": row.frames,
                "csi_loss_ratio": row.loss_ratio,
                "backend": "esp32",
            }
        )

    def start_session_meta(self) -> None:
        self._paths["session_dir"].mkdir(parents=True, exist_ok=True)
        meta = {
            "session_id": self.session_id,
            "project": self.project_dir.name,
            "created_at": now_iso(),
            "backend": "esp32",
            "schema": "presence.esp32.session.v1",
        }
        self._paths["session_json"].write_text(json.dumps(meta, indent=2))

    def write_parquet(self) -> Path:
        if not self._rows:
            raise ValueError("no windows recorded")
        df = pd.DataFrame(self._rows)
        out = self._paths["rf_windows_parquet"]
        df.to_parquet(out, index=False)
        self._write_labels_from_phases(df)
        return out

    def _write_labels_from_phases(self, rf: pd.DataFrame) -> None:
        rows = []
        for _, row in rf.iterrows():
            occ = _phase_to_occupancy(str(row.get("phase", "")))
            if occ is None:
                continue
            rows.append(
                {
                    "window_id": row["window_id"],
                    "occupancy": occ,
                    "label_source": "manual",
                    "phase": row.get("phase"),
                }
            )
        if rows:
            pd.DataFrame(rows).to_parquet(self._paths["labels_parquet"], index=False)


def _phase_to_occupancy(phase: str) -> Optional[str]:
    p = phase.lower()
    if "labeled_vacant" in p or p == "vacant":
        return "vacant"
    if "labeled_occupied" in p or "occupied" in p or "present" in p:
        return "occupied"
    return None


def record_session(
    repo_root: Path,
    project_dir: Path,
    session_id: str,
    *,
    duration_s: float,
    udp_port: int = 5005,
    phase: str = "calibration",
    window_seconds: float = 2.0,
) -> Path:
    recorder = Esp32SessionRecorder(project_dir, session_id, phase=phase)
    recorder.start_session_meta()

    def _on_window(row: Esp32WindowRow) -> None:
        recorder.on_window(row)

    svc = Esp32IngestService(
        repo_root,
        udp_port=udp_port,
        window_seconds=window_seconds,
        on_window=_on_window,
    )
    svc.start()
    try:
        time.sleep(max(0.1, duration_s))
    finally:
        svc.stop()
    return recorder.write_parquet()
