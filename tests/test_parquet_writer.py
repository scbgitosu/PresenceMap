"""Session parquet roundtrip."""
from __future__ import annotations

from pathlib import Path

from mac_app.train.dataset import (
    SessionParquetWriter,
    list_sessions,
    read_session_parquet,
    write_session_json,
)


def test_esp32_session_roundtrip(tmp_path: Path) -> None:
    project_dir = tmp_path / "p"
    write_session_json(project_dir, "s1", payload={})
    w = SessionParquetWriter(project_dir, "s1")
    w.write_rf_window(
        {
            "window_id": "w1",
            "ts_start": "2026-01-01T00:00:00Z",
            "ts_end": "2026-01-01T00:00:02Z",
            "phase": "labeled_vacant",
            "csi_features": [0.1, 0.2],
            "csi_frames": 10,
            "csi_loss_ratio": 0.0,
        }
    )
    w.write_label({"window_id": "w1", "occupancy": "vacant", "label_source": "manual"})
    w.finalize()
    assert "s1" in list_sessions(project_dir)
    rf = read_session_parquet(project_dir, "s1", kind="rf_windows")
    assert rf is not None and len(rf) == 1
