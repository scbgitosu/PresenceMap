"""ZMQ publisher sockets for the HP agent.

- ``publish_bind`` carries window + health messages (reliable enough; HWM=100).
- ``frames_bind`` carries multipart JPEG frames (CONFLATE on the subscriber so
  late frames drop in favor of fresh ones).
"""
from __future__ import annotations

from typing import Iterable

import zmq

from hp_agent.transport.messages import (
    TOPIC_HEALTH,
    TOPIC_WINDOW,
    encode,
)


class Publisher:
    def __init__(self, publish_bind: str, frames_bind: str) -> None:
        self.ctx = zmq.Context.instance()
        self.windows = self.ctx.socket(zmq.PUB)
        self.windows.setsockopt(zmq.SNDHWM, 100)
        self.windows.setsockopt(zmq.LINGER, 0)
        self.windows.bind(publish_bind)
        self.frames = self.ctx.socket(zmq.PUB)
        self.frames.setsockopt(zmq.SNDHWM, 20)
        self.frames.setsockopt(zmq.LINGER, 0)
        self.frames.bind(frames_bind)

    def send_window(self, payload: dict) -> None:
        self.windows.send_multipart([TOPIC_WINDOW, encode(payload)])

    def send_health(self, payload: dict) -> None:
        self.windows.send_multipart([TOPIC_HEALTH, encode(payload)])

    def send_frame_parts(self, parts: Iterable[bytes]) -> None:
        self.frames.send_multipart(list(parts))

    def close(self) -> None:
        try:
            self.windows.close(linger=0)
        except Exception:
            pass
        try:
            self.frames.close(linger=0)
        except Exception:
            pass
