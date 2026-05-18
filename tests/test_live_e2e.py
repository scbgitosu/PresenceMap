"""End-to-end live-mode test.

1. Build a synthetic session and train a small model on it.
2. Publish a stream of fresh WindowMsgs over ZMQ.
3. Run InferenceLoop against them; it should write predictions into the
   LiveBuffer.
4. Spin up the FastAPI app pointed at the same SQLite and hit /state,
   /history, /health.

Slow (~10 s) -- skip in tight loops with ``pytest -k "not test_live_e2e"``.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hp_agent.transport.messages import make_window_msg
from hp_agent.transport.zmq_publisher import Publisher
from mac_app.inference.live_buffer import LiveBuffer
from mac_app.inference.runtime import InferenceConfig, InferenceLoop
from mac_app.train.features import DatasetSpec
from mac_app.train.train import TrainConfig, train_session
from tests.test_train_eval import _make_session


def _ts(base: datetime, dt_s: float) -> str:
    return (base + timedelta(seconds=dt_s)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@pytest.mark.slow
def test_live_inference_writes_to_buffer_and_api(tmp_path: Path) -> None:
    # Train a model on synthetic data and force CPU so the test is portable.
    project_dir = _make_session(tmp_path, n_per_class=80)
    cfg = TrainConfig(epochs=20, batch_size=32, val_fraction=0.3, device="cpu", seed=11)
    meta = train_session(project_dir, ["synth001"], cfg=cfg)
    assert meta.eval_summary["accuracy"] > 0.85

    # Spin up a publisher.
    win_port = 5611
    frames_port = 5612
    pub = Publisher(f"tcp://127.0.0.1:{win_port}", f"tcp://127.0.0.1:{frames_port}")
    time.sleep(0.1)

    # Spin up the inference loop pointed at our test SQLite.
    sqlite_path = tmp_path / "live.sqlite3"
    buffer = LiveBuffer(sqlite_path)
    inf_cfg = InferenceConfig(
        project_dir=project_dir,
        model_id=meta.model_id,
        windows_endpoint=f"tcp://127.0.0.1:{win_port}",
        frames_endpoint=None,
        device="cpu",
    )
    loop = InferenceLoop(inf_cfg, buffer=buffer)
    loop.start()
    time.sleep(0.5)  # let SUB connect

    # Send 12 vacant windows then 12 occupied windows. Use feature vectors
    # similar to the training data so the model classifies them confidently.
    base = datetime(2026, 5, 18, 1, 0, 0, tzinfo=timezone.utc)
    import numpy as np
    from hp_agent.capture.csi_features import DEFAULT_NUM_SUBCARRIERS
    from tests.test_train_eval import _synth_window

    rng = np.random.default_rng(seed=33)
    for i, occupied in enumerate([False] * 12 + [True] * 12):
        msg = _synth_window(i, base, occupied=occupied, rng=rng, num_subcarriers=DEFAULT_NUM_SUBCARRIERS)
        # Overwrite session/window ids so they don't collide with the training set.
        msg["session_id"] = "live"
        msg["window_id"] = f"live_w{i:06d}"
        msg["phase"] = "live"
        pub.send_window(msg)
        time.sleep(0.08)
    time.sleep(0.6)

    # Verify predictions land in the buffer.
    history = buffer.predictions(limit=100)
    assert len(history) >= 20, f"expected at least 20 predictions, got {len(history)}"
    last = buffer.latest_prediction()
    assert last is not None and last.stable in ("vacant", "occupied")

    # The final ~12 windows are occupied; expect at least one stable=occupied row.
    occupied_rows = [r for r in history if r.stable == "occupied"]
    assert occupied_rows, "no row ever flipped to occupied"
    vacant_rows = [r for r in history if r.stable == "vacant"]
    assert vacant_rows, "no row was ever vacant"

    # Spin up FastAPI against the same SQLite + hit /state and /history.
    os.environ["PRESENCE_SQLITE_PATH"] = str(sqlite_path)
    os.environ["PRESENCE_PROJECT_DIR"] = str(project_dir)
    import importlib
    import mac_app.api.server as srv

    srv = importlib.reload(srv)
    from fastapi.testclient import TestClient

    with TestClient(srv.create_app()) as c:
        r = c.get("/state")
        assert r.status_code == 200
        body = r.json()
        assert body["model_id"] == meta.model_id
        r2 = c.get("/history", params={"limit": 100})
        assert r2.status_code == 200
        assert len(r2.json()["points"]) == len(history)
        r3 = c.get("/health")
        assert r3.status_code == 200
        assert r3.json()["active_model_id"] == meta.model_id

    loop.stop()
    pub.close()
    buffer.close()
