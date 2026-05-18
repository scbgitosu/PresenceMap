"""Per-window CSI feature aggregation.

Takes a list of ``CSIFrame``s collected within one window and produces a
fixed-length feature vector plus a downsampled amplitude matrix for offline
training.

Feature layout (see ``CSI_FEATURE_NAMES`` for the canonical order):

  amp_mean[K]         per-subcarrier amplitude mean         K=num_subcarriers
  amp_std[K]          per-subcarrier amplitude std
  phase_std[K]        per-subcarrier detrended-phase std
  amp_total_var       scalar amplitude variance across all subcarriers
  amp_pca_top1_eig    top eigenvalue of subcarrier covariance
  amp_pca_top1_ratio  top eigenvalue / sum of eigenvalues
  subc_corr_offdiag   mean of off-diagonal subcarrier correlation matrix
  temporal_acf_lag1   lag-1 ACF averaged across subcarriers
  csi_loss_ratio      1 - (actual_frames / expected_frames)

For HT20 K=56, so the vector is 56*3 + 6 = 174 features. The expected frame
count drives the loss ratio; the aggregator never silently fakes data when
frames are missing.

When fewer than 2 frames are present, statistics fall back to zeros and the
``loss_ratio`` is set to 1.0 so the downstream Mac-side trainer can decide
whether to drop the window.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from hp_agent.capture.csi_atheros import CSIFrame

DEFAULT_NUM_SUBCARRIERS = 56


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


def _frame_amplitude(frame: CSIFrame, num_subcarriers: int) -> np.ndarray:
    """Returns a (num_subcarriers,) amplitude vector, averaged across antenna pairs."""
    csi = frame.csi
    if csi.size == 0:
        return np.zeros(num_subcarriers, dtype=np.float32)
    amp = np.abs(csi)  # (nr, nc, num_tones)
    # Mean across all antenna pairs to flatten the antenna dims.
    amp_mean_ant = amp.reshape(-1, amp.shape[-1]).mean(axis=0)
    if amp_mean_ant.shape[0] >= num_subcarriers:
        return amp_mean_ant[:num_subcarriers].astype(np.float32)
    out = np.zeros(num_subcarriers, dtype=np.float32)
    out[: amp_mean_ant.shape[0]] = amp_mean_ant.astype(np.float32)
    return out


def _frame_phase(frame: CSIFrame, num_subcarriers: int) -> np.ndarray:
    """Returns per-subcarrier phase after antenna averaging."""
    csi = frame.csi
    if csi.size == 0:
        return np.zeros(num_subcarriers, dtype=np.float32)
    # Average complex values across antenna pairs *then* take angle so we don't
    # average phases directly (which is meaningless past +/- pi).
    csi_mean_ant = csi.reshape(-1, csi.shape[-1]).mean(axis=0)
    phase = np.angle(csi_mean_ant)
    if phase.shape[0] >= num_subcarriers:
        return phase[:num_subcarriers].astype(np.float32)
    out = np.zeros(num_subcarriers, dtype=np.float32)
    out[: phase.shape[0]] = phase.astype(np.float32)
    return out


def _detrend_phase(phase_matrix: np.ndarray) -> np.ndarray:
    """Subtract a per-frame linear fit (subcarrier index -> phase) to remove
    CFO/STO offsets that otherwise dominate phase variance.
    """
    n_frames, k = phase_matrix.shape
    if k < 2:
        return np.zeros_like(phase_matrix)
    # Unwrap along subcarrier axis to give the linear fit a sane sequence.
    unwrapped = np.unwrap(phase_matrix, axis=1)
    idx = np.arange(k, dtype=np.float32)
    # Fit y = a*x + b independently for each frame, vectorized.
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
    """Top eigenvalue + ratio of (top / sum) on the subcarrier covariance."""
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
    # Mask zero-variance subcarriers to avoid NaNs.
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
    """Lag-1 autocorrelation along time, averaged across subcarriers."""
    if amp_matrix.shape[0] < 3:
        return 0.0
    # Per-subcarrier zero-mean signal.
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
    frames: List[CSIFrame],
    *,
    expected_frames: int,
    num_subcarriers: int = DEFAULT_NUM_SUBCARRIERS,
    include_amp_matrix: bool = False,
) -> Tuple[List[float], Optional[bytes], float]:
    """Returns (feature_vector, amp_matrix_f16_bytes_or_None, loss_ratio).

    ``amp_matrix_f16_bytes_or_None`` is set when ``include_amp_matrix=True`` and
    we have at least one frame; the bytes are little-endian float16 in
    C-contiguous order with shape ``(n_frames, num_subcarriers)``. Mac decoders
    can reconstruct via ``np.frombuffer(buf, dtype='<f2').reshape(n, k)``.
    """
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
        + [
            amp_total_var,
            top_eig,
            top_ratio,
            subc_corr,
            acf1,
            loss_ratio,
        ]
    )

    amp_f16: Optional[bytes] = None
    if include_amp_matrix:
        amp_f16 = amp_matrix.astype(np.float16).tobytes()

    return feature_vector, amp_f16, loss_ratio
