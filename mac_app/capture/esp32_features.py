"""Per-window CSI features from ESP32 ADR-018 frames.

Uses ``mac_app.capture.csi_features`` so ESP32 windows feed ``TemporalCSIModel``
(174 CSI features for K=56).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mac_app.capture.csi_features import (
    DEFAULT_NUM_SUBCARRIERS,
    _detrend_phase,
    _pca_top_eigen,
    _subcarrier_corr_offdiag_mean,
    _temporal_acf_lag1,
    aggregate_window_features,
    csi_feature_dim,
    csi_feature_names,
)
from mac_app.capture.esp32_parser import Esp32CsiFrame

__all__ = [
    "DEFAULT_NUM_SUBCARRIERS",
    "csi_feature_names",
    "csi_feature_dim",
    "aggregate_esp32_window_features",
]


class _Esp32FrameAdapter:
    """Minimal adapter so we can reuse ``aggregate_window_features``."""

    def __init__(self, frame: Esp32CsiFrame, num_subcarriers: int) -> None:
        k = num_subcarriers
        amp = frame.amplitude
        if amp.shape[0] >= k:
            a = amp[:k]
        else:
            a = np.zeros(k, dtype=np.float32)
            a[: amp.shape[0]] = amp.astype(np.float32)
        phase = np.zeros(k, dtype=np.float32)
        self.csi = np.zeros((1, 1, k), dtype=np.complex64)
        for i in range(k):
            self.csi[0, 0, i] = complex(float(a[i]), 0.0)


def aggregate_esp32_window_features(
    frames: List[Esp32CsiFrame],
    *,
    expected_frames: int,
    num_subcarriers: int = DEFAULT_NUM_SUBCARRIERS,
    include_amp_matrix: bool = False,
) -> Tuple[List[float], Optional[bytes], float]:
    adapted = [_Esp32FrameAdapter(f, num_subcarriers) for f in frames]
    return aggregate_window_features(
        adapted,  # type: ignore[arg-type]
        expected_frames=expected_frames,
        num_subcarriers=num_subcarriers,
        include_amp_matrix=include_amp_matrix,
    )
