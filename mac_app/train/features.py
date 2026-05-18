"""Feature extraction + sliding-window dataset construction.

Turns ``rf_windows.parquet`` + ``labels.parquet`` for one or more sessions
into a fixed-shape ``(N, T, F)`` tensor of features plus matching label and
motion-intensity targets.

The feature vector is the per-window CSI features (``csi_features`` column)
concatenated with the RSSI features (scalars), plus simple per-window deltas
computed from a rolling history. Missing values become 0 (after scaling, this
sits at the dataset mean) and a per-feature ``valid`` mask flag isn't included
yet -- a future iteration can add it if missingness becomes informative.

The motion-intensity target is derived from the CSI ``amp_total_var`` feature
(``feats[3*K]`` where K = num_subcarriers) and normalized to [0, 1] by a fixed
log-scale; see ``MOTION_TARGET_SCALE``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from mac_app.train.dataset import read_session_parquet
from hp_agent.capture.csi_features import DEFAULT_NUM_SUBCARRIERS

# RSSI scalar columns from rf_windows.parquet (Stage 4 schema).
RSSI_FEATURE_COLS = [
    "target_rssi_avg_dbm",
    "target_rssi_std_db",
    "snr_db",
    "noise_dbm",
    "visible_bssid_count",
    "neighbor_rssi_sum_dbm",
    "channel_utilization_proxy",
    "target_seen",
]

# Rolling delta features (computed across consecutive rows of one session).
DELTA_FEATURE_COLS = [
    "rssi_delta_prev_db",
    "rssi_recent_std_db",
    "csi_amp_delta_prev",
    "csi_amp_recent_std",
]

# Motion-intensity target uses log1p(amp_total_var) / MOTION_TARGET_SCALE,
# clipped to [0, 1]. Tuned by hand against the test fixtures; revisit when
# we have real recordings.
MOTION_TARGET_SCALE = 6.0


@dataclass
class DatasetSpec:
    num_subcarriers: int = DEFAULT_NUM_SUBCARRIERS
    sequence_length: int = 8


@dataclass
class FeatureExtractor:
    """Holds the fitted scaler + feature ordering for a model."""

    spec: DatasetSpec
    feature_names: List[str]
    scaler: StandardScaler

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def feature_names(spec: DatasetSpec) -> List[str]:
    K = spec.num_subcarriers
    csi = (
        [f"amp_mean_{k:02d}" for k in range(K)]
        + [f"amp_std_{k:02d}" for k in range(K)]
        + [f"phase_std_{k:02d}" for k in range(K)]
        + [
            "amp_total_var",
            "amp_pca_top1_eig",
            "amp_pca_top1_ratio",
            "subc_corr_offdiag",
            "temporal_acf_lag1",
            "csi_loss_ratio",
        ]
    )
    return list(csi) + list(RSSI_FEATURE_COLS) + list(DELTA_FEATURE_COLS)


def _coerce_features_column(col: pd.Series, expected_dim: int) -> np.ndarray:
    """The ``csi_features`` column round-trips through parquet as either a
    Python list or a numpy object array. Stack into a (n_rows, expected_dim)
    float32 matrix, padding/truncating as needed.
    """
    arr = np.zeros((len(col), expected_dim), dtype=np.float32)
    for i, v in enumerate(col):
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float32)
        n = min(expected_dim, v.shape[0])
        arr[i, :n] = v[:n]
    return arr


def _rssi_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack RSSI scalar columns into a (n_rows, len(RSSI_FEATURE_COLS)) float32 matrix.

    Missing values become NaN; we replace with 0 after scaling.
    """
    out = np.zeros((len(df), len(RSSI_FEATURE_COLS)), dtype=np.float32)
    for j, col in enumerate(RSSI_FEATURE_COLS):
        if col not in df.columns:
            continue
        s = df[col]
        if s.dtype == bool:
            v = s.fillna(False).astype(np.float32).to_numpy()
        else:
            v = s.astype(float).fillna(0.0).to_numpy(dtype=np.float32)
        out[:, j] = v
    return out


def _delta_matrix(rssi: np.ndarray, csi: np.ndarray, K: int) -> np.ndarray:
    """Computes per-row rolling deltas. ``csi`` is the full csi_features matrix.

    Columns:
        rssi_delta_prev_db    -- target_rssi_avg_dbm[t] - target_rssi_avg_dbm[t-1]
        rssi_recent_std_db    -- target_rssi_avg_dbm std over last 5 rows
        csi_amp_delta_prev    -- amp_total_var[t] - amp_total_var[t-1]
        csi_amp_recent_std    -- amp_total_var std over last 5 rows
    """
    n = rssi.shape[0]
    rssi_avg = rssi[:, 0]   # target_rssi_avg_dbm column
    amp_total = csi[:, 3 * K] if csi.shape[1] >= 3 * K + 1 else np.zeros(n, dtype=np.float32)
    out = np.zeros((n, 4), dtype=np.float32)
    out[1:, 0] = rssi_avg[1:] - rssi_avg[:-1]
    out[1:, 2] = amp_total[1:] - amp_total[:-1]
    for i in range(n):
        lo = max(0, i - 4)
        if i - lo >= 1:
            out[i, 1] = float(rssi_avg[lo : i + 1].std())
            out[i, 3] = float(amp_total[lo : i + 1].std())
    return out


def extract_features_matrix(
    rf_windows: pd.DataFrame, spec: DatasetSpec
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X_raw, motion_target_raw) for one session, before scaling."""
    K = spec.num_subcarriers
    csi_dim = K * 3 + 6
    csi = _coerce_features_column(rf_windows["csi_features"], csi_dim)
    rssi = _rssi_matrix(rf_windows)
    deltas = _delta_matrix(rssi, csi, K)
    X = np.concatenate([csi, rssi, deltas], axis=1).astype(np.float32)
    amp_total = csi[:, 3 * K]
    motion = np.clip(np.log1p(np.maximum(amp_total, 0.0)) / MOTION_TARGET_SCALE, 0.0, 1.0).astype(np.float32)
    return X, motion


def fit_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    # Replace zero-variance columns' std with 1 so we don't blow up to inf.
    scaler.scale_ = np.where(scaler.scale_ < 1e-9, 1.0, scaler.scale_)
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X).astype(np.float32)


def build_sequences(
    X: np.ndarray, *, T: int
) -> np.ndarray:
    """Build a sliding-window tensor of shape (N, T, F).

    For row ``i`` the sequence is ``X[i-T+1 .. i]``, with left-padding
    repeating ``X[0]`` to avoid changing the row count.
    """
    n, f = X.shape
    if n == 0:
        return np.zeros((0, T, f), dtype=np.float32)
    pad = np.tile(X[0:1], (T - 1, 1))
    padded = np.concatenate([pad, X], axis=0)
    sequences = np.zeros((n, T, f), dtype=np.float32)
    for i in range(n):
        sequences[i] = padded[i : i + T]
    return sequences


@dataclass
class TrainingDataset:
    """Bundled training data after feature extraction + scaling + sequencing."""

    X: np.ndarray              # (N, T, F)
    y: np.ndarray              # (N,) int64: 0=vacant, 1=occupied
    motion: np.ndarray         # (N,) float32: motion intensity target in [0,1]
    window_ids: np.ndarray     # (N,) object: source window_id strings
    session_ids: np.ndarray    # (N,) object
    extractor: FeatureExtractor


def build_dataset(
    project_dir,
    session_ids: List[str],
    spec: Optional[DatasetSpec] = None,
    *,
    drop_missing_labels: bool = True,
) -> TrainingDataset:
    spec = spec or DatasetSpec()
    all_X = []
    all_motion = []
    all_y = []
    all_window_ids = []
    all_session_ids = []
    for sid in session_ids:
        rf = read_session_parquet(project_dir, sid, kind="rf_windows")
        lbl = read_session_parquet(project_dir, sid, kind="labels")
        if rf is None or rf.empty:
            continue
        if lbl is None:
            lbl = pd.DataFrame(columns=["window_id", "occupancy", "label_source"])
        merged = rf.merge(
            lbl[["window_id", "occupancy", "label_source"]],
            on="window_id",
            how="left",
        )
        if drop_missing_labels:
            merged = merged[merged["label_source"].isin(["yolo", "manual"])]
        if merged.empty:
            continue
        merged = merged.sort_values("ts_start").reset_index(drop=True)
        X_raw, motion = extract_features_matrix(merged, spec)
        y = (merged["occupancy"].fillna("vacant") == "occupied").astype(np.int64).to_numpy()
        all_X.append(X_raw)
        all_motion.append(motion)
        all_y.append(y)
        all_window_ids.append(merged["window_id"].to_numpy())
        all_session_ids.append(np.full(len(merged), sid, dtype=object))
    if not all_X:
        raise ValueError(f"no labeled windows across {session_ids!r}")
    X_concat = np.concatenate(all_X, axis=0)
    motion_concat = np.concatenate(all_motion, axis=0)
    y_concat = np.concatenate(all_y, axis=0)
    wid_concat = np.concatenate(all_window_ids, axis=0)
    sid_concat = np.concatenate(all_session_ids, axis=0)
    scaler = fit_scaler(X_concat)
    X_scaled = apply_scaler(X_concat, scaler)
    # Build sequences per-session so we don't bleed context across sessions.
    seq_parts = []
    cursor = 0
    for arr in all_X:
        n = arr.shape[0]
        seq_parts.append(build_sequences(X_scaled[cursor : cursor + n], T=spec.sequence_length))
        cursor += n
    X_seq = np.concatenate(seq_parts, axis=0)
    extractor = FeatureExtractor(
        spec=spec,
        feature_names=feature_names(spec),
        scaler=scaler,
    )
    return TrainingDataset(
        X=X_seq,
        y=y_concat,
        motion=motion_concat,
        window_ids=wid_concat,
        session_ids=sid_concat,
        extractor=extractor,
    )
