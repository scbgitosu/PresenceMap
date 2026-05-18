"""End-to-end training test using a synthetic v2 session.

Generates a session with rf_windows + labels parquet on disk, runs
``train_session``, asserts accuracy > 0.85 on the holdout, and reloads the
saved model from the registry.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from hp_agent.capture.csi_features import DEFAULT_NUM_SUBCARRIERS
from mac_app.train.dataset import SessionParquetWriter, write_session_json
from mac_app.train.features import DatasetSpec, build_dataset
from mac_app.train.registry import load_model
from mac_app.train.train import TrainConfig, train_session
from shared.versioning import SCHEMA_VERSION


def _ts(base: datetime, dt_s: float) -> str:
    return (base + timedelta(seconds=dt_s)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _synth_window(
    i: int, base: datetime, *, occupied: bool, rng: np.random.Generator, num_subcarriers: int
) -> dict:
    K = num_subcarriers
    # Per-subcarrier amplitudes: vacant ~ small constant, occupied ~ large + correlated motion.
    if occupied:
        base_amp = np.full(K, 120.0) + 25.0 * np.sin(2 * np.pi * i / 12.0)
        amp_mean = base_amp + rng.normal(0.0, 1.0, K)
        amp_std = np.full(K, 6.0) + rng.normal(0.0, 0.5, K)
        amp_total_var = 35.0 + rng.normal(0.0, 2.0)
        pca_top1_eig = 50.0 + rng.normal(0.0, 4.0)
        pca_top1_ratio = 0.45 + 0.02 * rng.standard_normal()
        subc_corr = 0.30 + 0.02 * rng.standard_normal()
        acf1 = 0.65 + 0.03 * rng.standard_normal()
        target_rssi = -52.0 + rng.normal(0.0, 1.5)
        snr = 42.0 + rng.normal(0.0, 1.5)
    else:
        amp_mean = np.full(K, 100.0) + rng.normal(0.0, 0.2, K)
        amp_std = np.full(K, 0.4) + rng.normal(0.0, 0.05, K)
        amp_total_var = 0.5 + 0.1 * rng.standard_normal()
        pca_top1_eig = 1.2 + 0.1 * rng.standard_normal()
        pca_top1_ratio = 0.10 + 0.01 * rng.standard_normal()
        subc_corr = -0.01 + 0.01 * rng.standard_normal()
        acf1 = 0.05 + 0.02 * rng.standard_normal()
        target_rssi = -56.0 + rng.normal(0.0, 0.5)
        snr = 39.0 + rng.normal(0.0, 0.5)
    phase_std = np.abs(rng.normal(0.0, 0.05, K))
    feats = (
        amp_mean.tolist()
        + amp_std.tolist()
        + phase_std.tolist()
        + [amp_total_var, pca_top1_eig, pca_top1_ratio, subc_corr, acf1, 0.0]
    )
    return {
        "schema": "presence.window.v2",
        "schema_version": SCHEMA_VERSION,
        "agent_id": "hp-test",
        "session_id": "sm",
        "window_id": f"sm_w{i:06d}",
        "ts_start": _ts(base, i * 2.0),
        "ts_end": _ts(base, i * 2.0 + 2.0),
        "phase": "labeled_occupied" if occupied else "labeled_vacant",
        "rssi": {
            "target_rssi_avg_dbm": float(target_rssi),
            "target_rssi_std_db": float(0.5 + 0.1 * rng.standard_normal()),
            "snr_db": float(snr),
            "noise_dbm": -95.0,
            "visible_bssid_count": 24,
            "neighbor_rssi_sum_dbm": -42.3,
            "channel_utilization_proxy": None,
            "target_seen": True,
        },
        "csi": {"features": feats, "frames": 60, "loss_ratio": 0.0},
        "interface": "wlan1",
        "backend": "iw_scan+csi",
    }


def _make_session(tmp_path: Path, session_id: str = "synth001", n_per_class: int = 100) -> Path:
    project_dir = tmp_path / "synth_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    write_session_json(project_dir, session_id, payload={"phases": [{"phase": "freeform"}]})
    w = SessionParquetWriter(project_dir, session_id)
    base = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(seed=0)
    idx = 0
    for occupied in (False, True):
        for _ in range(n_per_class):
            msg = _synth_window(idx, base, occupied=occupied, rng=rng, num_subcarriers=DEFAULT_NUM_SUBCARRIERS)
            w.write_window(msg)
            w.write_label({
                "window_id": msg["window_id"],
                "session_id": session_id,
                "ts_start": msg["ts_start"],
                "ts_end": msg["ts_end"],
                "occupancy": "occupied" if occupied else "vacant",
                "person_count": 1 if occupied else 0,
                "label_confidence": 0.95,
                "label_source": "yolo",
                "yolo_frame_refs": [],
                "yolo_model_id": "test",
                "note": "",
            })
            idx += 1
    w.finalize()
    return project_dir


def test_build_dataset_shapes(tmp_path: Path) -> None:
    project_dir = _make_session(tmp_path, n_per_class=30)
    ds = build_dataset(project_dir, ["synth001"], DatasetSpec(sequence_length=8))
    assert ds.X.ndim == 3
    assert ds.X.shape[1] == 8
    assert ds.X.shape[0] == 60      # 30 vacant + 30 occupied
    assert set(ds.y.tolist()) == {0, 1}
    assert ds.motion.shape == (60,)
    assert (ds.motion >= 0.0).all() and (ds.motion <= 1.0).all()


def test_end_to_end_train_and_reload(tmp_path: Path) -> None:
    project_dir = _make_session(tmp_path, n_per_class=80)
    cfg = TrainConfig(epochs=20, batch_size=32, val_fraction=0.3, device="cpu", seed=11)
    meta = train_session(project_dir, ["synth001"], cfg=cfg)
    assert meta.eval_summary["accuracy"] > 0.85, meta.eval_summary
    # Reload the saved weights and verify the head returns valid logits on a fresh sample.
    import torch

    model, extractor = load_model(project_dir, meta.model_id, device="cpu")
    ds = build_dataset(project_dir, ["synth001"], DatasetSpec(sequence_length=8))
    with torch.no_grad():
        logits, motion = model(torch.from_numpy(ds.X[:4]))
    assert logits.shape == (4, 2)
    assert motion.shape == (4,)
    # Reload feature names match training-time names.
    assert extractor.feature_names[:5] == ds.extractor.feature_names[:5]
