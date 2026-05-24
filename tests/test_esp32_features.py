"""Tests for ESP32 window feature aggregation."""
from __future__ import annotations

import time

from mac_app.capture.esp32_features import aggregate_esp32_window_features
from mac_app.capture.esp32_parser import Esp32CsiFrame, build_test_frame, parse_frame
import numpy as np


def _frame(node_id: int, seq: int) -> Esp32CsiFrame:
    pairs = [(i, (i + 1) % 20) for i in range(56)]
    data = build_test_frame(node_id, 1, pairs)
    f, _ = parse_frame(data)
    f.sequence = seq
    f.ts = time.time()
    return f


def test_aggregate_esp32_window_features_shape():
    frames = [_frame(1, i) for i in range(10)]
    feats, _, loss = aggregate_esp32_window_features(frames, expected_frames=40)
    assert len(feats) == 56 * 3 + 6
    assert 0.0 <= loss <= 1.0
