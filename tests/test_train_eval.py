"""End-to-end training on synthetic ESP32 session parquet."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from mac_app.capture.csi_features import DEFAULT_NUM_SUBCARRIERS, csi_feature_dim
from mac_app.train.dataset import SessionParquetWriter, write_session_json
from mac_app.train.features import DatasetSpec, build_dataset
from mac_app.train.registry import load_model
from mac_app.train.train import TrainConfig, train_session


def _ts(base: datetime, dt_s: float) -> str:
    return (base + timedelta(seconds=dt_s)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _csi_features(occupied: bool, i: int, rng: np.random.Generator) -> list:
    K = DEFAULT_NUM_SUBCARRIERS
    if occupied:
        amp_mean = np.full(K, 120.0) + 25.0 * np.sin(2 * np.pi * i / 12.0) + rng.normal(0, 1, K)
        amp_std = np.full(K, 6.0)
        phase_std = np.full(K, 0.3)
        scalars = [35.0, 50.0, 0.45, 0.30, 0.65, 0.0]
    else:
        amp_mean = np.full(K, 100.0) + rng.normal(0, 0.2, K)
        amp_std = np.full(K, 0.4)
        phase_std = np.full(K, 0.05)
        scalars = [0.5, 1.2, 0.10, -0.01, 0.05, 0.0]
    return amp_mean.tolist() + amp_std.tolist() + phase_std.tolist() + scalars


def _make_session(tmp_path: Path, session_id: str = "synth001", n_per_class: int = 80) -> Path:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    write_session_json(project_dir, session_id, payload={"backend": "esp32"})
    w = SessionParquetWriter(project_dir, session_id)
    base = datetime(2026, 5, 18, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    idx = 0
    for occupied in (False, True):
        for _ in range(n_per_class):
            wid = str(uuid.uuid4())
            phase = "labeled_occupied" if occupied else "labeled_vacant"
            w.write_rf_window(
                {
                    "window_id": wid,
                    "ts_start": _ts(base, idx * 2.0),
                    "ts_end": _ts(base, idx * 2.0 + 2.0),
                    "phase": phase,
                    "csi_features": _csi_features(occupied, idx, rng),
                    "csi_frames": 40,
                    "csi_loss_ratio": 0.0,
                }
            )
            w.write_label(
                {
                    "window_id": wid,
                    "occupancy": "occupied" if occupied else "vacant",
                    "label_source": "manual",
                }
            )
            idx += 1
    w.finalize()
    return project_dir


def test_build_dataset_shapes(tmp_path: Path) -> None:
    project_dir = _make_session(tmp_path, n_per_class=30)
    ds = build_dataset(project_dir, ["synth001"], DatasetSpec(sequence_length=8))
    assert ds.X.shape[1] == 8
    assert ds.X.shape[2] == csi_feature_dim() + 2
    assert ds.X.shape[0] == 60


def test_end_to_end_train_and_reload(tmp_path: Path) -> None:
    project_dir = _make_session(tmp_path, n_per_class=80)
    meta = train_session(
        project_dir,
        ["synth001"],
        cfg=TrainConfig(epochs=20, batch_size=32, val_fraction=0.3, device="cpu", seed=11),
    )
    assert meta.eval_summary["accuracy"] > 0.85
    import torch

    model, extractor = load_model(project_dir, meta.model_id, device="cpu")
    ds = build_dataset(project_dir, ["synth001"], DatasetSpec(sequence_length=8))
    with torch.no_grad():
        logits, motion = model(torch.from_numpy(ds.X[:4]))
    assert logits.shape == (4, 2)
