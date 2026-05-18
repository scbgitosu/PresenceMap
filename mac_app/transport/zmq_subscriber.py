"""ZMQ subscriber sockets for the Mac side.

Two sockets:
- ``windows_connect`` subscribes to TOPIC_WINDOW + TOPIC_HEALTH.
- ``frames_connect`` subscribes to TOPIC_FRAME with CONFLATE so only the
  freshest JPEG is delivered.

The subscriber's ``poll_messages`` is non-blocking and returns whatever's
available; the caller decides how often to drain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import zmq

from hp_agent.transport.messages import (
    TOPIC_FRAME,
    TOPIC_HEALTH,
    TOPIC_WINDOW,
    decode,
    parse_frame_parts,
)


@dataclass
class WindowEnvelope:
    topic: str  # "window" | "health"
    payload: dict


class Subscriber:
    def __init__(
        self,
        windows_connect: str,
        frames_connect: Optional[str] = None,
        *,
        recv_hwm: int = 1000,
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.windows = self.ctx.socket(zmq.SUB)
        self.windows.setsockopt(zmq.RCVHWM, recv_hwm)
        self.windows.setsockopt(zmq.LINGER, 0)
        self.windows.connect(windows_connect)
        self.windows.setsockopt(zmq.SUBSCRIBE, TOPIC_WINDOW)
        self.windows.setsockopt(zmq.SUBSCRIBE, TOPIC_HEALTH)
        self.frames: Optional[zmq.Socket] = None
        if frames_connect:
            # NB: ZMQ_CONFLATE is incompatible with multipart messages, so we
            # emulate "deliver only the freshest" with RCVHWM=1.
            self.frames = self.ctx.socket(zmq.SUB)
            self.frames.setsockopt(zmq.RCVHWM, 1)
            self.frames.setsockopt(zmq.LINGER, 0)
            self.frames.connect(frames_connect)
            self.frames.setsockopt(zmq.SUBSCRIBE, TOPIC_FRAME)

    def poll_windows(self, timeout_ms: int = 0) -> List[WindowEnvelope]:
        out: List[WindowEnvelope] = []
        while True:
            try:
                parts = self.windows.recv_multipart(flags=zmq.NOBLOCK if timeout_ms == 0 else 0)
            except zmq.Again:
                break
            if len(parts) != 2:
                continue
            topic, buf = parts
            payload = decode(buf)
            label = "window" if topic == TOPIC_WINDOW else ("health" if topic == TOPIC_HEALTH else topic.decode("utf-8", "replace"))
            out.append(WindowEnvelope(topic=label, payload=payload))
            if timeout_ms:
                timeout_ms = 0  # only block once
        return out

    def poll_frame(self) -> Optional[dict]:
        if self.frames is None:
            return None
        try:
            parts = self.frames.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        if len(parts) != 4:
            return None
        return parse_frame_parts(parts)

    def close(self) -> None:
        try:
            self.windows.close(linger=0)
        except Exception:
            pass
        if self.frames is not None:
            try:
                self.frames.close(linger=0)
            except Exception:
                pass
