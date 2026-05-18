"""Tests for CSI binary parsing and feature aggregation."""
from __future__ import annotations

import numpy as np

from hp_agent.capture.csi_atheros import (
    CSIFrame,
    build_csi_frame_bytes,
    parse_csi_frame,
    pack_csi_payload,
    unpack_csi_payload,
)
from hp_agent.capture.csi_features import (
    DEFAULT_NUM_SUBCARRIERS,
    aggregate_window_features,
    csi_feature_dim,
    csi_feature_names,
)


def test_csi_payload_roundtrip() -> None:
    rng = np.random.default_rng(seed=42)
    nr, nc, num_tones = 1, 1, 56
    # Stay inside the 10-bit signed range [-512, 511].
    real = rng.integers(-400, 401, size=(nr, nc, num_tones))
    imag = rng.integers(-400, 401, size=(nr, nc, num_tones))
    csi = (real + 1j * imag).astype(np.complex64)
    payload = pack_csi_payload(csi)
    rebuilt = unpack_csi_payload(payload, nr=nr, nc=nc, num_tones=num_tones)
    assert rebuilt.shape == csi.shape
    np.testing.assert_array_equal(rebuilt, csi)


def test_csi_frame_header_roundtrip() -> None:
    csi = np.zeros((1, 1, 56), dtype=np.complex64)
    csi[0, 0, 0] = 12 + 7j
    csi[0, 0, 55] = -34 - 5j
    blob = build_csi_frame_bytes(
        tstamp_ns=1234567890,
        nr=1, nc=1, num_tones=56,
        csi=csi,
        bandwidth=0,
        rssi=212, rssi_1=212, rssi_2=0, rssi_3=0,
        noise_dbm=-95, rate=7, tx_channel=2437,
    )
    frame, off = parse_csi_frame(blob, 0)
    assert off == len(blob)
    assert frame.nr == 1 and frame.nc == 1 and frame.num_tones == 56
    assert frame.noise_dbm == -95
    assert frame.rssi == 212
    assert frame.tstamp_ns == 1234567890
    assert complex(frame.csi[0, 0, 0]) == 12 + 7j
    assert complex(frame.csi[0, 0, 55]) == -34 - 5j


def _synth_frame(amp_values: np.ndarray, *, num_tones: int = DEFAULT_NUM_SUBCARRIERS) -> CSIFrame:
    csi = np.zeros((1, 1, num_tones), dtype=np.complex64)
    csi[0, 0, : amp_values.shape[0]] = amp_values.astype(np.complex64)
    return CSIFrame(
        ts=0.0, csi=csi,
        nr=1, nc=1, num_tones=num_tones,
        bandwidth=0,
        rssi=200, rssi_1=200, rssi_2=0, rssi_3=0,
        noise_dbm=-95, rate=7, tstamp_ns=0,
    )


def test_feature_vector_shape() -> None:
    assert csi_feature_dim() == DEFAULT_NUM_SUBCARRIERS * 3 + 6
    assert len(csi_feature_names()) == csi_feature_dim()
    # 60 vacant frames (no motion -> tiny variance)
    frames = []
    for _ in range(60):
        amps = np.full(DEFAULT_NUM_SUBCARRIERS, 100.0) + np.random.default_rng(0).normal(0, 0.5, DEFAULT_NUM_SUBCARRIERS)
        frames.append(_synth_frame(amps))
    feats, amp_blob, loss = aggregate_window_features(
        frames, expected_frames=60, include_amp_matrix=True
    )
    assert len(feats) == csi_feature_dim()
    assert loss == 0.0
    assert amp_blob is not None
    # blob shape: (60, 56) float16
    arr = np.frombuffer(amp_blob, dtype="<f2").reshape(60, DEFAULT_NUM_SUBCARRIERS)
    assert arr.shape == (60, DEFAULT_NUM_SUBCARRIERS)
    # csi_loss_ratio is the last feature and should be 0.0 here
    assert feats[-1] == 0.0


def test_feature_vector_motion_increases_variance() -> None:
    rng = np.random.default_rng(seed=7)
    K = DEFAULT_NUM_SUBCARRIERS
    # Vacant: low-noise constant amplitude.
    vacant = [_synth_frame(np.full(K, 100.0) + rng.normal(0, 0.2, K)) for _ in range(60)]
    # Occupied (moving person): amplitude swings + correlated subcarriers.
    occupied = []
    for t in range(60):
        base = np.full(K, 100.0) + 15.0 * np.sin(2 * np.pi * t / 10.0)
        noise = rng.normal(0, 1.0, K)
        occupied.append(_synth_frame(base + noise))
    feats_v, _, _ = aggregate_window_features(vacant, expected_frames=60)
    feats_o, _, _ = aggregate_window_features(occupied, expected_frames=60)
    # amp_total_var is feature index = 3*K (first scalar)
    var_idx = 3 * K
    assert feats_o[var_idx] > feats_v[var_idx] * 10
    # top1 eigenvalue (3*K + 1) should be way higher under motion
    assert feats_o[var_idx + 1] > feats_v[var_idx + 1] * 10
    # subc_corr_offdiag (3*K + 3) should be much higher under correlated motion
    assert feats_o[var_idx + 3] > feats_v[var_idx + 3]


def test_feature_loss_ratio() -> None:
    # 30 of 60 expected = 0.5 loss
    frames = [_synth_frame(np.full(DEFAULT_NUM_SUBCARRIERS, 50.0)) for _ in range(30)]
    feats, _, loss = aggregate_window_features(frames, expected_frames=60)
    assert abs(loss - 0.5) < 1e-9
    assert feats[-1] == loss


def test_feature_empty_window_returns_placeholders() -> None:
    feats, blob, loss = aggregate_window_features([], expected_frames=60)
    assert len(feats) == csi_feature_dim()
    assert blob is None
    assert loss == 1.0
    assert feats[-1] == 1.0
