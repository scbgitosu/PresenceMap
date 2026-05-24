"""Tests for ESP32 ADR-018 parser."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from mac_app.capture.esp32_parser import (
    ESP32_CSI_MAGIC,
    RUVIEW_VITALS_MAGIC,
    ParseError,
    ParseErrorKind,
    build_test_frame,
    parse_frame,
    parse_stream,
    ruview_sibling_packet_name,
)


def test_parse_valid_frame_56_subcarriers():
    pairs = [(i, (i * 2) % 127) for i in range(56)]
    data = build_test_frame(1, 1, pairs)
    frame, consumed = parse_frame(data)
    assert consumed == 20 + 56 * 2
    assert frame.node_id == 1
    assert frame.n_antennas == 1
    assert frame.n_subcarriers == 56
    assert frame.rssi_dbm == -50
    assert frame.subcarrier_count == 56
    assert frame.amplitude[0] == pytest.approx(0.0, abs=0.01)
    assert frame.amplitude[1] == pytest.approx((1**2 + 2**2) ** 0.5, rel=0.01)


def test_insufficient_data():
    with pytest.raises(ParseError) as exc:
        parse_frame(b"\x00" * 10)
    assert exc.value.kind == ParseErrorKind.INSUFFICIENT_DATA


def test_invalid_magic():
    data = build_test_frame(1, 1, [(10, 20)])
    buf = bytearray(data)
    struct.pack_into("<I", buf, 0, 0xDEADBEEF)
    with pytest.raises(ParseError) as exc:
        parse_frame(bytes(buf))
    assert exc.value.kind == ParseErrorKind.INVALID_MAGIC


def test_sibling_vitals_packet():
    data = bytearray(32)
    struct.pack_into("<I", data, 0, RUVIEW_VITALS_MAGIC)
    data = bytes(data)
    with pytest.raises(ParseError) as exc:
        parse_frame(data)
    assert exc.value.kind == ParseErrorKind.NON_CSI_PACKET
    assert ruview_sibling_packet_name(RUVIEW_VITALS_MAGIC)


def test_parse_stream_two_frames():
    pairs = [(10 + i, 20 + i) for i in range(4)]
    f1 = build_test_frame(1, 1, pairs)
    f2 = build_test_frame(2, 1, pairs)
    combined = f1 + f2
    frames, _ = parse_stream(combined)
    assert len(frames) == 2
    assert frames[0].node_id == 1
    assert frames[1].node_id == 2


def test_parse_stream_resync_after_garbage():
    pairs = [(10 + i, 20 + i) for i in range(4)]
    frame = build_test_frame(1, 1, pairs)
    data = bytes([0xFF, 0xFF, 0xFF]) + frame
    frames, _ = parse_stream(data)
    assert len(frames) == 1


def test_multi_antenna():
    pairs = []
    for ant in range(3):
        for sc in range(4):
            pairs.append((ant * 10 + sc, (ant * 10 + sc) * 2))
    data = build_test_frame(5, 3, pairs)
    frame, consumed = parse_frame(data)
    assert consumed == 20 + 12 * 2
    assert frame.node_id == 5
    assert frame.n_antennas == 3
    assert frame.n_subcarriers == 4
    assert frame.amplitude.shape == (4,)


def test_amplitude_known_iq():
    data = build_test_frame(1, 1, [(100, 0), (0, 50), (30, 40)])
    frame, _ = parse_frame(data)
    assert frame.amplitude[0] == pytest.approx(100.0, rel=0.01)
    assert frame.amplitude[1] == pytest.approx(50.0, rel=0.01)
    assert frame.amplitude[2] == pytest.approx(50.0, rel=0.01)
