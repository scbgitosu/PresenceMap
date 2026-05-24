"""Live inference from ESP32 fused window features in runtime state."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import torch

from mac_app.capture.csi_features import csi_feature_dim
from mac_app.inference.live_buffer import LiveBuffer, PredictionRow
from mac_app.inference.state_smoother import StateSmoother
from mac_app.train.features import DatasetSpec, apply_scaler, build_sequences
from mac_app.train.registry import load_model
from shared.project import runtime_dir


@dataclass
class Esp32InferenceConfig:
    project_dir: Path
    model_id: str
    repo_root: Path
    device: str = "cpu"
    poll_sleep_s: float = 0.5
    sequence_length: int = 8


class Esp32InferenceLoop:
    def __init__(self, cfg: Esp32InferenceConfig, buffer: Optional[LiveBuffer] = None) -> None:
        self.cfg = cfg
        self.buffer = buffer or LiveBuffer()
        self.model, self.extractor = load_model(cfg.project_dir, cfg.model_id, device=cfg.device)
        self.model.eval()
        self.smoother = StateSmoother(window_size=5, threshold=3)
        self._csi_history: Deque[np.ndarray] = deque(maxlen=cfg.sequence_length + 4)
        self._feat_history: Deque[np.ndarray] = deque(maxlen=cfg.sequence_length)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_fused_ts: float = 0.0
        self._state_path = runtime_dir(cfg.repo_root) / "esp32_state.json"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="esp32-inference", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._tick()
            time.sleep(self.cfg.poll_sleep_s)

    def _append_deltas(self, csi_vec: np.ndarray) -> np.ndarray:
        K = self.extractor.spec.num_subcarriers
        amp_idx = 3 * K
        amp_total = float(csi_vec[amp_idx]) if csi_vec.shape[0] > amp_idx else 0.0
        delta_prev = 0.0
        if self._csi_history:
            prev = self._csi_history[-1]
            prev_amp = float(prev[amp_idx]) if prev.shape[0] > amp_idx else 0.0
            delta_prev = amp_total - prev_amp
        recent = [float(v[amp_idx]) for v in self._csi_history if v.shape[0] > amp_idx]
        recent.append(amp_total)
        recent_std = float(np.std(recent)) if len(recent) > 1 else 0.0
        return np.concatenate([csi_vec, np.array([delta_prev, recent_std], dtype=np.float32)])

    def _tick(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text())
        except Exception:
            return
        fused = data.get("fused_window")
        if not fused or not fused.get("features"):
            return
        ts = float(fused.get("ts_end", 0))
        if ts <= self._last_fused_ts:
            return
        self._last_fused_ts = ts
        csi_vec = np.asarray(fused["features"], dtype=np.float32)
        if csi_vec.shape[0] != csi_feature_dim(self.extractor.spec.num_subcarriers):
            return
        self._csi_history.append(csi_vec)
        row = self._append_deltas(csi_vec)
        if row.shape[0] != self.extractor.n_features:
            return
        self._feat_history.append(row)
        if len(self._feat_history) < self.cfg.sequence_length:
            return
        mat = np.stack(list(self._feat_history), axis=0)
        scaled = apply_scaler(mat, self.extractor.scaler)
        seq = build_sequences(scaled, T=self.cfg.sequence_length)[-1:]
        x = torch.from_numpy(seq).to(self.cfg.device)
        with torch.no_grad():
            occ_logits, motion = self.model(x)
        probs = torch.softmax(occ_logits, dim=-1)[0]
        conf = float(probs.max().item())
        raw = "occupied" if int(probs.argmax()) == 1 else "vacant"
        stable = self.smoother.update(raw)
        mot = float(motion[0].item())
        self.buffer.write_prediction(
            PredictionRow(
                window_id=fused.get("window_id", "esp32-fused"),
                raw=raw,
                stable=stable,
                confidence=conf,
                motion_intensity=mot,
                model_id=self.cfg.model_id,
                received_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        )
