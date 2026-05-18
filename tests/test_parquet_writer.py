"""Tests for the v2 session parquet writer + reader roundtrip."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mac_app.train.dataset import (
    SessionParquetWriter,
    read_session_json,
    read_session_parquet,
    write_session_json,
)
from shared.versioning import SCHEMA_VERSION


def _window(window_id: str, *, snr: float | None, target_seen: bool, csi_frames: int) -> dict:
    return {
        "schema": "presence.window.v2",
        "schema_version": SCHEMA_VERSION,
        "agent_id": "hp-test",
        "session_id": "sm",
        "window_id": window_id,
        "ts_start": "2026-05-18T00:00:00.000Z",
        "ts_end": "2026-05-18T00:00:02.000Z",
        "phase": "labeled_vacant",
        "rssi": {
            "target_rssi_avg_dbm": -55.5,
            "target_rssi_std_db": 1.2,
            "snr_db": snr,
            "noise_dbm": -95.0,
            "visible_bssid_count": 24,
            "neighbor_rssi_sum_dbm": -42.3,
            "channel_utilization_proxy": None,
            "target_seen": target_seen,
        },
        "csi": {
            "features": [0.1, 0.2, 0.3, 0.4],
            "frames": csi_frames,
            "loss_ratio": 0.0,
        },
        "interface": "wlan1",
        "backend": "iw_scan",
    }


def test_session_json_roundtrip(tmp_path: Path) -> None:
    write_session_json(tmp_path, "sess001", payload={
        "session_id": "sess001",
        "phases": [{"phase": "freeform", "ts_start": "t0"}],
    })
    out = read_session_json(tmp_path, "sess001")
    assert out["session_id"] == "sess001"
    assert out["schema_version"] == SCHEMA_VERSION


def test_parquet_writer_roundtrip(tmp_path: Path) -> None:
    w = SessionParquetWriter(tmp_path, "sess001")
    for i, target_seen in enumerate([True, False, True, True]):
        w.write_window(_window(f"w{i:03d}", snr=40.0 if target_seen else None, target_seen=target_seen, csi_frames=60))
        w.write_health({
            "schema": "presence.health.v2",
            "schema_version": SCHEMA_VERSION,
            "agent_id": "hp-test",
            "session_id": "sess001",
            "ts": "2026-05-18T00:00:00.000Z",
            "code": "ok",
            "detail": "heartbeat",
            "metrics": {"target_seen": target_seen, "loss_ratio": 0.0},
        })
        w.write_label({
            "window_id": f"w{i:03d}",
            "session_id": "sess001",
            "ts_start": "t0",
            "ts_end": "t1",
            "occupancy": "occupied" if target_seen else "vacant",
            "person_count": 1 if target_seen else 0,
            "label_confidence": 0.9,
            "label_source": "yolo",
            "yolo_frame_refs": [f"f{i:03d}_a", f"f{i:03d}_b"],
            "yolo_model_id": "yolov8n.pt",
            "note": "",
        })
    w.write_yolo_frame({
        "frame_id": "f000_a",
        "ts": 1700000000_000_000,
        "session_id": "sess001",
        "agent_id": "hp-test",
        "jpeg_path": "thumbnails/f000_a.jpg",
        "person_count": 1,
        "max_conf": 0.91,
        "bboxes": [{"cls": 0, "conf": 0.91, "x1": 1.0, "y1": 2.0, "x2": 100.0, "y2": 200.0}],
        "yolo_model_id": "yolov8n.pt",
    })
    counts_before = w.counts()
    assert counts_before == {"rf_windows": 4, "labels": 4, "yolo_frames": 1, "health": 4, "thumbnails": 0}

    w.finalize()

    rf = read_session_parquet(tmp_path, "sess001", kind="rf_windows")
    labels = read_session_parquet(tmp_path, "sess001", kind="labels")
    health = read_session_parquet(tmp_path, "sess001", kind="health")
    yolo = read_session_parquet(tmp_path, "sess001", kind="yolo_frames")
    thumbs = read_session_parquet(tmp_path, "sess001", kind="thumbnails")
    assert rf is not None and len(rf) == 4
    assert labels is not None and len(labels) == 4
    assert health is not None and len(health) == 4
    assert yolo is not None and len(yolo) == 1
    assert thumbs is None  # we never wrote thumbnails
    # Schema fidelity:
    assert rf.iloc[0]["snr_db"] == 40.0
    assert bool(rf.iloc[1]["target_seen"]) is False
    # snr null preserved as NaN (or None)
    snr_row_1 = rf.iloc[1]["snr_db"]
    assert snr_row_1 is None or (isinstance(snr_row_1, float) and np.isnan(snr_row_1))
    assert rf.iloc[0]["csi_frames"] == 60
    assert labels.iloc[0]["label_source"] == "yolo"
    assert list(labels.iloc[0]["yolo_frame_refs"]) == ["f000_a", "f000_b"]
    assert int(yolo.iloc[0]["bboxes"][0]["conf"] * 100) == 91
