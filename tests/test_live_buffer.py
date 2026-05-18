"""Tests for the SQLite-backed LiveBuffer."""
from __future__ import annotations

from pathlib import Path

from mac_app.inference.live_buffer import LiveBuffer, PredictionRow


def _row(window_id: str, ts: str, *, stable: str = "occupied", conf: float = 0.9) -> PredictionRow:
    return PredictionRow(
        window_id=window_id,
        session_id="s",
        ts_start=ts,
        ts_end=ts,
        raw=stable,
        stable=stable,
        confidence=conf,
        motion_intensity=0.5,
        model_id="m1",
        received_at=ts,
    )


def test_prediction_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite3"
    buf = LiveBuffer(db)
    buf.write_prediction(_row("w001", "2026-05-18T00:00:01.000Z"))
    buf.write_prediction(_row("w002", "2026-05-18T00:00:03.000Z", stable="vacant"))
    latest = buf.latest_prediction()
    assert latest is not None
    assert latest.window_id == "w002"
    history = buf.predictions(limit=10)
    assert [r.window_id for r in history] == ["w001", "w002"]
    buf.close()


def test_prediction_since_filter(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite3"
    buf = LiveBuffer(db)
    buf.write_prediction(_row("w001", "2026-05-18T00:00:01.000Z"))
    buf.write_prediction(_row("w002", "2026-05-18T00:00:03.000Z"))
    buf.write_prediction(_row("w003", "2026-05-18T00:00:05.000Z"))
    history = buf.predictions(since="2026-05-18T00:00:02.000Z", limit=10)
    assert [r.window_id for r in history] == ["w002", "w003"]


def test_health_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite3"
    buf = LiveBuffer(db)
    buf.write_health({
        "ts": "2026-05-18T00:00:01.000Z",
        "agent_id": "hp-1",
        "session_id": "s",
        "code": "target_ssid_low_visibility",
        "detail": "visible in 30%",
        "metrics": {"hit_rate": 0.3},
    })
    rows = buf.recent_health(limit=10)
    assert len(rows) == 1 and rows[0]["code"] == "target_ssid_low_visibility"


def test_meta_kv(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite3"
    buf = LiveBuffer(db)
    assert buf.get_meta("active_model_id") is None
    buf.set_meta("active_model_id", "m1")
    assert buf.get_meta("active_model_id") == "m1"
    buf.set_meta("active_model_id", "m2")
    assert buf.get_meta("active_model_id") == "m2"
