"""Live inference loop: subscribe to an HP agent, score windows, persist state.

Mirrors the ``SessionRecorder`` design but for the live path: no parquet
writing, no YOLO -- just feature extraction + model prediction + state
smoothing + LiveBuffer writes. The Streamlit Live page owns one of these in
``st.session_state`` so start/stop is per-browser-session, and the FastAPI
server reads the same SQLite file.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np
import pandas as pd
import torch

from mac_app.inference.live_buffer import LiveBuffer, PredictionRow
from mac_app.inference.state_smoother import StateSmoother
from mac_app.train.features import (
    DatasetSpec,
    apply_scaler,
    extract_features_matrix,
)
from mac_app.train.registry import load_model
from mac_app.transport.reconnect import Watchdog
from mac_app.transport.zmq_subscriber import Subscriber
from shared.utils import now_iso


@dataclass
class InferenceConfig:
    project_dir: Path
    model_id: str
    windows_endpoint: str = "tcp://localhost:5555"
    frames_endpoint: Optional[str] = None
    device: str = "cpu"
    poll_sleep_s: float = 0.05
    stall_threshold_s: float = 6.0
    history_buffer: int = 8


@dataclass
class InferenceStats:
    windows: int = 0
    healths: int = 0
    last_state: str = "unknown"
    last_confidence: float = 0.0
    last_motion: float = 0.0
    last_window_age_s: float = float("inf")
    signal_lost: bool = True


class InferenceLoop:
    """Loads a model + subscribes to the agent + runs predictions per window."""

    def __init__(self, cfg: InferenceConfig, buffer: Optional[LiveBuffer] = None) -> None:
        self.cfg = cfg
        self.buffer = buffer or LiveBuffer()
        self.model, self.extractor = load_model(cfg.project_dir, cfg.model_id, device=cfg.device)
        self.model.eval()
        self.smoother = StateSmoother(window_size=5, threshold=3)
        self._sub: Optional[Subscriber] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._raw_history: Deque[np.ndarray] = deque(maxlen=self.extractor.spec.sequence_length)
        self._stats = InferenceStats()
        self._watchdog = Watchdog(stall_threshold_s=cfg.stall_threshold_s)

    # ---- public API ------------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sub = Subscriber(self.cfg.windows_endpoint, frames_connect=self.cfg.frames_endpoint)
        self.buffer.set_meta("active_model_id", self.cfg.model_id)
        self.buffer.set_meta("inference_started_at", now_iso())
        self._thread = threading.Thread(target=self._run, name="inference_loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._sub is not None:
            self._sub.close()
            self._sub = None
        self.buffer.set_meta("inference_stopped_at", now_iso())

    def stats(self) -> InferenceStats:
        with self._lock:
            self._stats.last_window_age_s = self._watchdog.age_s()
            self._stats.signal_lost = self._watchdog.is_signal_lost()
            return InferenceStats(**self._stats.__dict__)

    # ---- main loop -------------------------------------------------------- #

    def _run(self) -> None:
        assert self._sub is not None
        while not self._stop.is_set():
            envs = self._sub.poll_windows()
            for env in envs:
                if env.topic == "window":
                    self._on_window(env.payload)
                elif env.topic == "health":
                    self.buffer.write_health(env.payload)
                    with self._lock:
                        self._stats.healths += 1
            time.sleep(self.cfg.poll_sleep_s)

    def _on_window(self, msg: Dict[str, Any]) -> None:
        # Build a single-row DataFrame the feature extractor understands.
        df = pd.DataFrame([_flatten_window_for_features(msg)])
        try:
            X_raw, motion_target = extract_features_matrix(df, self.extractor.spec)
        except Exception:
            return
        if X_raw.shape[0] == 0:
            return
        # Need a feature vector whose dimensionality matches the model.
        if X_raw.shape[1] != self.extractor.n_features:
            # Pad / truncate to the model's expected width.
            row = np.zeros(self.extractor.n_features, dtype=np.float32)
            n = min(X_raw.shape[1], self.extractor.n_features)
            row[:n] = X_raw[0, :n]
            X_raw_padded = row[None, :]
        else:
            X_raw_padded = X_raw
        X_scaled = apply_scaler(X_raw_padded, self.extractor.scaler)
        feat = X_scaled[0]  # (F,)
        with self._lock:
            self._raw_history.append(feat)
            if len(self._raw_history) < self.extractor.spec.sequence_length:
                # Bootstrap: left-pad with the first sample.
                pad = [self._raw_history[0]] * (
                    self.extractor.spec.sequence_length - len(self._raw_history)
                )
                seq = np.stack(pad + list(self._raw_history), axis=0)
            else:
                seq = np.stack(list(self._raw_history), axis=0)
        with torch.no_grad():
            logits, motion = self.model(
                torch.from_numpy(seq[None, :, :].astype(np.float32)).to(self.cfg.device)
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            motion_pred = float(motion.cpu().numpy()[0])
        raw_label = "occupied" if int(probs.argmax()) == 1 else "vacant"
        confidence = float(probs.max())
        ts = msg.get("ts_end") or msg.get("ts_start") or now_iso()
        smoothed = self.smoother.update(raw_label, confidence, motion_pred, ts=ts)
        self._watchdog.mark_received()
        row = PredictionRow(
            window_id=msg.get("window_id", ""),
            session_id=msg.get("session_id", ""),
            ts_start=msg.get("ts_start", ""),
            ts_end=msg.get("ts_end", ""),
            raw=raw_label,
            stable=smoothed.stable,
            confidence=confidence,
            motion_intensity=motion_pred,
            model_id=self.cfg.model_id,
            received_at=now_iso(),
        )
        self.buffer.write_prediction(row)
        with self._lock:
            self._stats.windows += 1
            self._stats.last_state = smoothed.stable
            self._stats.last_confidence = confidence
            self._stats.last_motion = motion_pred


def _flatten_window_for_features(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter: turn a wire WindowMsg into the column layout that
    ``mac_app.train.features.extract_features_matrix`` expects.
    """
    rssi = msg.get("rssi", {}) or {}
    csi = msg.get("csi", {}) or {}
    row: Dict[str, Any] = {
        "window_id": msg.get("window_id"),
        "session_id": msg.get("session_id"),
        "ts_start": msg.get("ts_start"),
        "ts_end": msg.get("ts_end"),
        "csi_features": list(csi.get("features", []) or []),
        "csi_frames": int(csi.get("frames", 0) or 0),
        "csi_loss_ratio": float(csi.get("loss_ratio", 1.0) or 1.0),
    }
    # RSSI scalars (StandardScaler will null-handle NaNs via fillna in features.py)
    for k in (
        "target_rssi_avg_dbm",
        "target_rssi_std_db",
        "snr_db",
        "noise_dbm",
        "visible_bssid_count",
        "neighbor_rssi_sum_dbm",
        "channel_utilization_proxy",
        "target_seen",
    ):
        row[k] = rssi.get(k)
    return row
