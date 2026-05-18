"""Tests for the YOLO-frame -> RF-window label aligner."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mac_app.yolo.labeler import WindowKey, YoloObservation, label_windows


def _ts_at(base: datetime, dt_s: float) -> str:
    return (base + timedelta(seconds=dt_s)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def test_label_alignment_basic() -> None:
    base = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc)
    windows = [
        WindowKey(window_id="w001", session_id="s", ts_start=_ts_at(base, 0), ts_end=_ts_at(base, 2)),
        WindowKey(window_id="w002", session_id="s", ts_start=_ts_at(base, 2), ts_end=_ts_at(base, 4)),
        WindowKey(window_id="w003", session_id="s", ts_start=_ts_at(base, 4), ts_end=_ts_at(base, 6)),
    ]
    # Frames at: 0.5 (vacant), 1.5 (vacant), 2.5 (occupied conf=0.8),
    # 3.5 (occupied conf=0.9), no frame in window 3.
    base_us = int(base.timestamp() * 1_000_000)
    obs = [
        YoloObservation(frame_id="f001", ts_us=base_us + 500_000, person_count=0, max_conf=0.0),
        YoloObservation(frame_id="f002", ts_us=base_us + 1_500_000, person_count=0, max_conf=0.0),
        YoloObservation(frame_id="f003", ts_us=base_us + 2_500_000, person_count=1, max_conf=0.8),
        YoloObservation(frame_id="f004", ts_us=base_us + 3_500_000, person_count=1, max_conf=0.9),
    ]
    labels = label_windows(windows, obs, yolo_model_id="yolov8n.pt")
    assert len(labels) == 3
    assert labels[0].occupancy == "vacant" and labels[0].label_source == "yolo"
    assert labels[0].yolo_frame_refs == ["f001", "f002"]
    assert labels[1].occupancy == "occupied"
    assert labels[1].person_count == 1
    assert abs(labels[1].label_confidence - 0.85) < 1e-6
    assert labels[2].label_source == "missing"
    assert labels[2].yolo_frame_refs == []


def test_label_manual_override_wins_over_yolo() -> None:
    base = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc)
    windows = [
        WindowKey(window_id="w001", session_id="s", ts_start=_ts_at(base, 0), ts_end=_ts_at(base, 2))
    ]
    base_us = int(base.timestamp() * 1_000_000)
    obs = [
        YoloObservation(frame_id="f001", ts_us=base_us + 500_000, person_count=1, max_conf=0.9)
    ]
    labels = label_windows(
        windows, obs, yolo_model_id="yolov8n.pt",
        override={"w001": "vacant"},
    )
    assert labels[0].label_source == "manual"
    assert labels[0].occupancy == "vacant"
    assert labels[0].person_count == 0
    # The yolo frame ref is preserved for traceability even on override.
    assert labels[0].yolo_frame_refs == ["f001"]
