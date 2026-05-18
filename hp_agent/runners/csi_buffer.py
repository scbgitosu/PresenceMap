"""Thread-safe buffer of CSI frames feeding the window loop.

A background thread reads frames from a ``CSISource`` and ``append``s them
here; the window loop ``drain``s them at each window boundary.
"""
from __future__ import annotations

import threading
from typing import List, Optional

from hp_agent.capture.csi_atheros import CSIFrame


class CSIBuffer:
    def __init__(self, max_frames: int = 1024) -> None:
        self._lock = threading.Lock()
        self._frames: List[CSIFrame] = []
        self._max = max_frames

    def append(self, frame: CSIFrame) -> None:
        with self._lock:
            if len(self._frames) >= self._max:
                self._frames.pop(0)
            self._frames.append(frame)

    def drain(self) -> List[CSIFrame]:
        with self._lock:
            out = self._frames
            self._frames = []
        return out

    def peek_count(self) -> int:
        with self._lock:
            return len(self._frames)


class CSICaptureThread:
    """Runs a ``CSISource`` in a daemon thread, pushing frames into a buffer."""

    def __init__(self, source, buffer: CSIBuffer) -> None:
        self._source = source
        self._buffer = buffer
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="csi_capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for frame in self._source.frames():
                if self._stop.is_set():
                    break
                self._buffer.append(frame)
        finally:
            try:
                self._source.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        try:
            self._source.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
