"""Feature extraction for ESP32 CSI training sessions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mac_app.capture.csi_features import DEFAULT_NUM_SUBCARRIERS, csi_feature_dim, csi_feature_names
from mac_app.train.dataset import read_session_parquet

DELTA_FEATURE_COLS = ["csi_amp_delta_prev", "csi_amp_recent_std"]
MOTION_TARGET_SCALE = 6.0


@dataclass
class DatasetSpec:
    num_subcarriers: int = DEFAULT_NUM_SUBCARRIERS
    sequence_length: int = 8


@dataclass
class FeatureExtractor:
    spec: DatasetSpec
    feature_names: List[str]
    scaler: StandardScaler

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _labels_from_esp32_phases(rf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in rf.iterrows():
        phase = str(row.get("phase", "")).lower()
        occ = None
        if "labeled_vacant" in phase or phase == "vacant":
            occ = "vacant"
        elif "labeled_occupied" in phase or "occupied" in phase or "present" in phase:
            occ = "occupied"
        if occ is None:
            continue
        rows.append(
            {
                "window_id": row["window_id"],
                "occupancy": occ,
                "label_source": "esp32_phase",
            }
        )
    return pd.DataFrame(rows)


def feature_names(spec: DatasetSpec) -> List[str]:
    return list(csi_feature_names(spec.num_subcarriers)) + list(DELTA_FEATURE_COLS)


def _coerce_features_column(col: pd.Series, expected_dim: int) -> np.ndarray:
    arr = np.zeros((len(col), expected_dim), dtype=np.float32)
    for i, v in enumerate(col):
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float32)
        n = min(expected_dim, v.shape[0])
        arr[i, :n] = v[:n]
    return arr


def _delta_matrix(csi: np.ndarray, K: int) -> np.ndarray:
    n = csi.shape[0]
    amp_total = csi[:, 3 * K] if csi.shape[1] >= 3 * K + 1 else np.zeros(n, dtype=np.float32)
    out = np.zeros((n, 2), dtype=np.float32)
    out[1:, 0] = amp_total[1:] - amp_total[:-1]
    for i in range(n):
        lo = max(0, i - 4)
        if i - lo >= 1:
            out[i, 1] = float(amp_total[lo : i + 1].std())
    return out


def extract_features_matrix(
    rf_windows: pd.DataFrame, spec: DatasetSpec
) -> Tuple[np.ndarray, np.ndarray]:
    K = spec.num_subcarriers
    csi_dim = csi_feature_dim(K)
    csi = _coerce_features_column(rf_windows["csi_features"], csi_dim)
    deltas = _delta_matrix(csi, K)
    X = np.concatenate([csi, deltas], axis=1).astype(np.float32)
    amp_total = csi[:, 3 * K]
    motion = np.clip(np.log1p(np.maximum(amp_total, 0.0)) / MOTION_TARGET_SCALE, 0.0, 1.0).astype(
        np.float32
    )
    return X, motion


def fit_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    scaler.scale_ = np.where(scaler.scale_ < 1e-9, 1.0, scaler.scale_)
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X).astype(np.float32)


def build_sequences(X: np.ndarray, *, T: int) -> np.ndarray:
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
    X: np.ndarray
    y: np.ndarray
    motion: np.ndarray
    window_ids: np.ndarray
    session_ids: np.ndarray
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
        if lbl.empty and "phase" in rf.columns:
            lbl = _labels_from_esp32_phases(rf)
        merged = rf.merge(
            lbl[["window_id", "occupancy", "label_source"]],
            on="window_id",
            how="left",
        )
        if drop_missing_labels:
            merged = merged[merged["label_source"].isin(["manual", "esp32_phase"])]
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
