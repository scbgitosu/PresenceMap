"""Session recorder: subscribe to an HP agent and write a training session.

The Streamlit Training/Collect page builds one of these and calls ``start()``
/ ``stop()``. The recorder runs a background drain loop:

- Window messages append to the in-memory ``SessionParquetWriter``.
- Health messages append to the same writer.
- Frames are decoded, YOLO runs on them, the metadata + (rate-limited)
  thumbnail land in the writer, and the labeler aligns observations to the
  current set of windows to produce a label per window.

The labels are recomputed each tick so manual phase overrides apply
immediately. On ``stop()`` we finalize parquet, run the thumbnail purge, and
write the final ``session.json`` with calibration stats.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2  # type: ignore[import-not-found]
import numpy as np

from mac_app.transport.reconnect import Watchdog
from mac_app.transport.zmq_subscriber import Subscriber
from mac_app.train.dataset import SessionParquetWriter, write_session_json
from mac_app.yolo.detector import YoloDetector, summarize
from mac_app.yolo.labeler import WindowKey, YoloObservation, label_windows
from mac_app.yolo.thumbnails import ThumbnailWriter, purge_session
from shared.utils import now_iso
from shared.versioning import SCHEMA_VERSION


@dataclass
class RecorderStats:
    windows: int = 0
    healths: int = 0
    frames: int = 0
    persons_seen_max: int = 0
    last_window_age_s: float = float("inf")
    last_yolo_person_count: int = 0
    last_yolo_conf: float = 0.0
    signal_lost: bool = True


@dataclass
class RecorderConfig:
    project_dir: Path
    session_id: str
    windows_endpoint: str = "tcp://localhost:5555"
    frames_endpoint: str = "tcp://localhost:5556"
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    yolo_min_interval_s: float = 0.5  # ~2 Hz inference cadence
    enable_yolo: bool = True
    thumbnail_fps: float = 1.0
    poll_sleep_s: float = 0.05
    stall_threshold_s: float = 6.0
    initial_phase: str = "freeform"

    # Computed
    overrides: Dict[str, str] = field(default_factory=dict)


class SessionRecorder:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self.writer = SessionParquetWriter(cfg.project_dir, cfg.session_id)
        self.thumbnails = ThumbnailWriter(self.writer.paths["session_dir"])
        self._sub: Optional[Subscriber] = None
        self._yolo: Optional[YoloDetector] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._windows: List[WindowKey] = []
        self._observations: List[YoloObservation] = []
        self._stats = RecorderStats()
        self._watchdog = Watchdog(stall_threshold_s=cfg.stall_threshold_s)
        self._phase = cfg.initial_phase
        self._last_yolo_us: int = 0
        self._yolo_model_id: str = ""

    # ---- public API ------------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None:
            return
        write_session_json(
            self.cfg.project_dir,
            self.cfg.session_id,
            payload={
                "session_id": self.cfg.session_id,
                "project": str(self.cfg.project_dir),
                "created_at": now_iso(),
                "ended_at": None,
                "phases": [{"phase": self._phase, "ts_start": now_iso()}],
                "calibration_stats": {},
                "schema_version": SCHEMA_VERSION,
            },
        )
        self._sub = Subscriber(self.cfg.windows_endpoint, frames_connect=self.cfg.frames_endpoint)
        if self.cfg.enable_yolo:
            self._yolo = YoloDetector(self.cfg.yolo_model, conf_threshold=self.cfg.yolo_conf)
            self._yolo_model_id = self._yolo.model_id
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._sub is not None:
            self._sub.close()
            self._sub = None
        # Final relabel + write labels parquet.
        self._recompute_labels()
        self.writer.finalize()
        # Update session.json with end time + counts.
        payload = {
            "session_id": self.cfg.session_id,
            "project": str(self.cfg.project_dir),
            "created_at": None,
            "ended_at": now_iso(),
            "phases": [{"phase": self._phase}],
            "calibration_stats": {},
            "counts": self.writer.counts(),
            "schema_version": SCHEMA_VERSION,
        }
        write_session_json(self.cfg.project_dir, self.cfg.session_id, payload=payload)
        # Purge thumbnails (rate + state-change).
        snap = self.writer.snapshot()["thumbnails"].to_dict("records")
        if snap:
            purge_session(self.writer.paths["session_dir"], thumbnails_index=snap)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def override_window(self, window_id: str, occupancy: str) -> None:
        with self._lock:
            self.cfg.overrides[window_id] = occupancy

    def stats(self) -> RecorderStats:
        with self._lock:
            self._stats.last_window_age_s = self._watchdog.age_s()
            self._stats.signal_lost = self._watchdog.is_signal_lost()
            return RecorderStats(**self._stats.__dict__)

    def snapshot(self) -> Dict[str, Any]:
        return self.writer.snapshot()

    # ---- main loop -------------------------------------------------------- #

    def _run(self) -> None:
        assert self._sub is not None
        while not self._stop.is_set():
            envs = self._sub.poll_windows()
            for env in envs:
                if env.topic == "window":
                    self._on_window(env.payload)
                elif env.topic == "health":
                    self.writer.write_health(env.payload)
                    with self._lock:
                        self._stats.healths += 1
            frame = self._sub.poll_frame()
            if frame is not None:
                self._on_frame(frame)
            time.sleep(self.cfg.poll_sleep_s)

    def _on_window(self, msg: Dict[str, Any]) -> None:
        # Stamp the current phase onto incoming windows so a mid-session
        # phase flip on the dashboard takes effect immediately.
        with self._lock:
            msg["phase"] = self._phase
            self._windows.append(
                WindowKey(
                    window_id=msg["window_id"],
                    session_id=msg["session_id"],
                    ts_start=msg["ts_start"],
                    ts_end=msg["ts_end"],
                    phase=msg["phase"],
                )
            )
            self._stats.windows += 1
        self.writer.write_window(msg)
        self._watchdog.mark_received()
        # Recompute labels for the just-arrived window using whatever YOLO
        # observations we have so far. (Late YOLO obs will be folded in at
        # the next window or at stop.)
        self._recompute_labels()

    def _on_frame(self, frame: Dict[str, Any]) -> None:
        ts_us = int(frame["ts_us"])
        jpeg = frame["jpeg"]
        try:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return
        if img is None:
            return
        frame_id = f"f{ts_us}"
        person_count = 0
        max_conf = 0.0
        detections = []
        run_yolo = self.cfg.enable_yolo and (
            (ts_us - self._last_yolo_us) >= int(self.cfg.yolo_min_interval_s * 1_000_000)
        )
        if run_yolo:
            try:
                detections = self._yolo.detect(img) if self._yolo else []
            except Exception:
                detections = []
            person_count, max_conf = summarize(detections)
            self._last_yolo_us = ts_us
            with self._lock:
                self._stats.last_yolo_person_count = person_count
                self._stats.last_yolo_conf = max_conf
                if person_count > self._stats.persons_seen_max:
                    self._stats.persons_seen_max = person_count
                self._observations.append(
                    YoloObservation(
                        frame_id=frame_id,
                        ts_us=ts_us,
                        person_count=person_count,
                        max_conf=max_conf,
                    )
                )
        # Always store a yolo frames row -- "missing" detection is itself a signal.
        thumb_path: Optional[Path] = None
        result = self.thumbnails.maybe_write(img, ts_us, frame_id=frame_id)
        if result is not None:
            thumb_path = result[0]
            self.writer.write_thumbnail_index(
                {
                    "frame_id": frame_id,
                    "ts": ts_us,
                    "path": str(thumb_path.relative_to(self.writer.paths["session_dir"])),
                    "person_count": person_count,
                    "kept_reason": "rate",
                }
            )
        self.writer.write_yolo_frame(
            {
                "frame_id": frame_id,
                "ts": ts_us,
                "session_id": self.cfg.session_id,
                "agent_id": frame.get("agent_id"),
                "jpeg_path": str(thumb_path.relative_to(self.writer.paths["session_dir"])) if thumb_path else None,
                "person_count": person_count,
                "max_conf": max_conf,
                "bboxes": [d.to_dict() for d in detections],
                "yolo_model_id": self._yolo_model_id,
            }
        )
        with self._lock:
            self._stats.frames += 1

    def _recompute_labels(self) -> None:
        with self._lock:
            windows_snapshot = list(self._windows)
            obs_snapshot = list(self._observations)
            overrides = dict(self.cfg.overrides)
        labeled = label_windows(
            windows_snapshot,
            obs_snapshot,
            yolo_model_id=self._yolo_model_id,
            override=overrides,
        )
        # Replace label buffer wholesale (cheap for hundreds of rows).
        with self.writer._lock:  # type: ignore[attr-defined]
            self.writer._labels = [lw.to_row() for lw in labeled]
