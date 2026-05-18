"""Webcam frame producer loop.

Spins up a ``WebcamProducer`` thread and routes its JPEG output to the ZMQ
frames socket. Health events (``camera_lost`` / ``ok``) are forwarded to the
``HeartbeatEmitter`` so the Mac side sees them in the same stream as RF.
"""
from __future__ import annotations

import logging

from hp_agent.capture.webcam_stream import WebcamConfig, WebcamProducer
from hp_agent.config import AgentConfig
from hp_agent.health.heartbeat import HeartbeatEmitter
from hp_agent.transport.messages import make_frame_parts
from hp_agent.transport.zmq_publisher import Publisher
from hp_agent.util.logging import get_logger


class FrameLoop:
    def __init__(
        self,
        cfg: AgentConfig,
        publisher: Publisher,
        heartbeat: HeartbeatEmitter,
        *,
        session_id: str,
    ) -> None:
        self.cfg = cfg
        self.publisher = publisher
        self.heartbeat = heartbeat
        self.session_id = session_id
        self.log = get_logger("hp_agent.frame_loop")
        self.log.setLevel(logging.INFO)
        wcfg = WebcamConfig(
            index_hint=cfg.webcam_index_hint,
            fps=cfg.webcam_fps,
            jpeg_quality=cfg.webcam_jpeg_quality,
            max_width=cfg.webcam_max_width,
            max_height=cfg.webcam_max_height,
        )
        self.producer = WebcamProducer(
            wcfg,
            on_jpeg=self._on_jpeg,
            on_health=self._on_health,
        )

    def start(self) -> None:
        if not self.cfg.webcam_enabled:
            self.log.info("webcam disabled in config")
            return
        self.producer.start()

    def stop(self) -> None:
        self.producer.stop()

    def _on_jpeg(self, jpeg: bytes, ts_us: int) -> None:
        parts = make_frame_parts(agent_id=self.cfg.agent_id, ts_us=ts_us, jpeg=jpeg)
        self.publisher.send_frame_parts(parts)

    def _on_health(self, code: str, detail: str) -> None:
        self.heartbeat.emit(code, detail=detail, source="webcam")
