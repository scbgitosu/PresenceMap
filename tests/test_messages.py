"""Roundtrip tests for the msgpack wire schemas."""
from __future__ import annotations

from hp_agent.transport import messages as m


def test_window_msg_roundtrip() -> None:
    rssi = {
        "target_rssi_avg_dbm": -55.5,
        "target_rssi_std_db": 1.2,
        "snr_db": 40.0,
        "noise_dbm": -95.0,
        "visible_bssid_count": 24,
        "neighbor_rssi_sum_dbm": -42.3,
        "channel_utilization_proxy": None,
        "target_seen": True,
    }
    win = m.make_window_msg(
        agent_id="hp-01",
        session_id="train001",
        window_id="train001_w000001",
        ts_start="2026-05-17T00:00:00.000Z",
        ts_end="2026-05-17T00:00:02.000Z",
        phase="labeled_vacant",
        rssi=rssi,
        interface="wlan1",
        backend="iw_scan",
    )
    buf = m.encode(win)
    out = m.decode(buf)
    assert out["schema"] == m.WINDOW_SCHEMA
    assert out["rssi"]["target_seen"] is True
    assert out["rssi"]["snr_db"] == 40.0
    assert out["csi"]["frames"] == 0  # placeholder for Stage 3
    assert out["window_id"] == "train001_w000001"


def test_health_msg_roundtrip() -> None:
    h = m.make_health_msg(
        agent_id="hp-01",
        session_id="train001",
        ts="2026-05-17T00:00:01.000Z",
        code="target_ssid_low_visibility",
        detail="visible in 30% of last 60 windows",
        metrics={"hit_rate": 0.3},
    )
    out = m.decode(m.encode(h))
    assert out["code"] == "target_ssid_low_visibility"
    assert out["metrics"]["hit_rate"] == 0.3


def test_frame_parts_roundtrip() -> None:
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg"
    parts = list(m.make_frame_parts(agent_id="hp-01", ts_us=1234567890, jpeg=jpeg))
    assert parts[0] == m.TOPIC_FRAME
    parsed = m.parse_frame_parts(parts)
    assert parsed["agent_id"] == "hp-01"
    assert parsed["ts_us"] == 1234567890
    assert parsed["jpeg"] == jpeg


def test_control_msg() -> None:
    c = m.make_control_msg("start_session", {"session_id": "train001", "phase": "calibration"})
    out = m.decode(m.encode(c))
    assert out["cmd"] == "start_session"
    assert out["args"]["phase"] == "calibration"
