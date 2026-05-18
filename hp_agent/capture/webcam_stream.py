"""Webcam JPEG producer.

Resolves a working camera index at startup (tries the hint, then 0..3), then
runs at a fixed FPS encoding frames to JPEG. If the camera goes away mid-run
we emit ``camera_lost`` health events and try to reopen every 30 s — never
silently faking labels.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2  # type: ignore[import-not-found]


@dataclass
class WebcamConfig:
    index_hint: int = 0
    fps: float = 5.0
    jpeg_quality: int = 75
    max_width: int = 640
    max_height: int = 480


def resolve_camera_index(hint: int) -> Optional[int]:
    """Try ``hint`` first, then 0..3. Returns the first that opens, else None."""
    candidates = [hint] + [i for i in (0, 1, 2, 3) if i != hint]
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            return idx
    return None


def _open_capture(idx: int, cfg: WebcamConfig) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.max_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.max_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    return cap


class WebcamProducer:
    """Background thread that grabs frames and dispatches JPEGs.

    The dispatch callback receives (jpeg_bytes, ts_us). Heartbeat callbacks are
    invoked on open success and every persistent failure so the surrounding
    agent can emit a ``camera_lost`` HealthMsg.
    """

    def __init__(
        self,
        cfg: WebcamConfig,
        *,
        on_jpeg: Callable[[bytes, int], None],
        on_health: Callable[[str, str], None],
    ) -> None:
        self.cfg = cfg
        self.on_jpeg = on_jpeg
        self.on_health = on_health
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._index: Optional[int] = None

    @property
    def resolved_index(self) -> Optional[int]:
        return self._index

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="webcam", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        period = 1.0 / max(self.cfg.fps, 0.1)
        cap: Optional[cv2.VideoCapture] = None
        last_health = 0.0
        retry_interval = 30.0
        while not self._stop.is_set():
            if cap is None:
                idx = resolve_camera_index(self.cfg.index_hint)
                if idx is None:
                    now = time.monotonic()
                    if now - last_health > retry_interval:
                        self.on_health("camera_lost", "no working camera index")
                        last_health = now
                    self._stop.wait(retry_interval)
                    continue
                cap = _open_capture(idx, self.cfg)
                if cap is None:
                    self.on_health("camera_lost", f"index {idx} opened then failed to configure")
                    cap = None
                    self._stop.wait(retry_interval)
                    continue
                self._index = idx
                self.on_health("ok", f"webcam ready on index {idx}")
            t0 = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                self.on_health("camera_lost", "read() returned no frame")
                cap.release()
                cap = None
                self._stop.wait(retry_interval)
                continue
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])
            if not ok2:
                self.on_health("camera_lost", "jpeg encode failed")
                continue
            ts_us = time.time_ns() // 1000
            self.on_jpeg(bytes(buf), ts_us)
            elapsed = time.monotonic() - t0
            remaining = period - elapsed
            if remaining > 0:
                self._stop.wait(remaining)
        if cap is not None:
            cap.release()
