"""Per-window CSI feature aggregation (ESP32 / complex CSI tensors)."""
from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

import numpy as np

DEFAULT_NUM_SUBCARRIERS = 56


class CsiFrameLike(Protocol):
    csi: np.ndarray


def csi_feature_names(num_subcarriers: int = DEFAULT_NUM_SUBCARRIERS) -> List[str]:
    names = [f"amp_mean_{k:02d}" for k in range(num_subcarriers)]
    names += [f"amp_std_{k:02d}" for k in range(num_subcarriers)]
    names += [f"phase_std_{k:02d}" for k in range(num_subcarriers)]
    names += [
        "amp_total_var",
        "amp_pca_top1_eig",
        "amp_pca_top1_ratio",
        "subc_corr_offdiag",
        "temporal_acf_lag1",
        "csi_loss_ratio",
    ]
    return names


def csi_feature_dim(num_subcarriers: int = DEFAULT_NUM_SUBCARRIERS) -> int:
    return num_subcarriers * 3 + 6


def _frame_amplitude(frame: CsiFrameLike, num_subcarriers: int) -> np.ndarray:
    csi = frame.csi
    if csi.size == 0:
        return np.zeros(num_subcarriers, dtype=np.float32)
    amp = np.abs(csi)
    amp_mean_ant = amp.reshape(-1, amp.shape[-1]).mean(axis=0)
    if amp_mean_ant.shape[0] >= num_subcarriers:
        return amp_mean_ant[:num_subcarriers].astype(np.float32)
    out = np.zeros(num_subcarriers, dtype=np.float32)
    out[: amp_mean_ant.shape[0]] = amp_mean_ant.astype(np.float32)
    return out


def _frame_phase(frame: CsiFrameLike, num_subcarriers: int) -> np.ndarray:
    csi = frame.csi
    if csi.size == 0:
        return np.zeros(num_subcarriers, dtype=np.float32)
    csi_mean_ant = csi.reshape(-1, csi.shape[-1]).mean(axis=0)
    phase = np.angle(csi_mean_ant)
    if phase.shape[0] >= num_subcarriers:
        return phase[:num_subcarriers].astype(np.float32)
    out = np.zeros(num_subcarriers, dtype=np.float32)
    out[: phase.shape[0]] = phase.astype(np.float32)
    return out


def _detrend_phase(phase_matrix: np.ndarray) -> np.ndarray:
    n_frames, k = phase_matrix.shape
    if k < 2:
        return np.zeros_like(phase_matrix)
    unwrapped = np.unwrap(phase_matrix, axis=1)
    idx = np.arange(k, dtype=np.float32)
    x_mean = idx.mean()
    x_var = float(np.sum((idx - x_mean) ** 2))
    if x_var == 0.0:
        return np.zeros_like(phase_matrix)
    y_mean = unwrapped.mean(axis=1, keepdims=True)
    a = (unwrapped - y_mean) @ (idx - x_mean) / x_var
    a = a[:, None]
    b = y_mean - a * x_mean
    fitted = a * idx + b
    return (unwrapped - fitted).astype(np.float32)


def _pca_top_eigen(amp_matrix: np.ndarray) -> Tuple[float, float]:
    if amp_matrix.shape[0] < 2:
        return 0.0, 0.0
    cov = np.cov(amp_matrix, rowvar=False)
    if cov.ndim == 0:
        return float(cov), 1.0
    eigvals = np.linalg.eigvalsh(cov)
    total = float(eigvals.sum())
    top = float(eigvals[-1]) if eigvals.size else 0.0
    if total <= 0:
        return top, 0.0
    return top, top / total


def _subcarrier_corr_offdiag_mean(amp_matrix: np.ndarray) -> float:
    if amp_matrix.shape[0] < 2 or amp_matrix.shape[1] < 2:
        return 0.0
    stds = amp_matrix.std(axis=0)
    valid = stds > 1e-9
    if valid.sum() < 2:
        return 0.0
    sub = amp_matrix[:, valid]
    corr = np.corrcoef(sub, rowvar=False)
    if corr.ndim == 0:
        return float(corr)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(corr[mask].mean())


def _temporal_acf_lag1(amp_matrix: np.ndarray) -> float:
    if amp_matrix.shape[0] < 3:
        return 0.0
    centered = amp_matrix - amp_matrix.mean(axis=0, keepdims=True)
    num = (centered[:-1] * centered[1:]).sum(axis=0)
    den = (centered ** 2).sum(axis=0)
    safe = den > 1e-12
    if not safe.any():
        return 0.0
    acf = np.zeros_like(num)
    acf[safe] = num[safe] / den[safe]
    return float(acf[safe].mean())


def aggregate_window_features(
    frames: List[CsiFrameLike],
    *,
    expected_frames: int,
    num_subcarriers: int = DEFAULT_NUM_SUBCARRIERS,
    include_amp_matrix: bool = False,
) -> Tuple[List[float], Optional[bytes], float]:
    expected = max(1, int(expected_frames))
    n = len(frames)
    loss_ratio = float(max(0.0, 1.0 - (n / expected)))

    if n == 0:
        zeros = [0.0] * (num_subcarriers * 3)
        scalars = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return zeros + scalars, None, 1.0

    amp_matrix = np.stack([_frame_amplitude(f, num_subcarriers) for f in frames], axis=0)
    phase_matrix = np.stack([_frame_phase(f, num_subcarriers) for f in frames], axis=0)
    phase_detrended = _detrend_phase(phase_matrix)

    amp_mean = amp_matrix.mean(axis=0)
    amp_std = amp_matrix.std(axis=0)
    phase_std = phase_detrended.std(axis=0)

    amp_total_var = float(amp_matrix.var())
    top_eig, top_ratio = _pca_top_eigen(amp_matrix)
    subc_corr = _subcarrier_corr_offdiag_mean(amp_matrix)
    acf1 = _temporal_acf_lag1(amp_matrix)

    feature_vector = (
        amp_mean.tolist()
        + amp_std.tolist()
        + phase_std.tolist()
        + [amp_total_var, top_eig, top_ratio, subc_corr, acf1, loss_ratio]
    )

    amp_f16: Optional[bytes] = None
    if include_amp_matrix:
        amp_f16 = amp_matrix.astype(np.float16).tobytes()

    return feature_vector, amp_f16, loss_ratio
