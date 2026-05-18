"""Agent configuration: resolve from project_config.json + CLI overrides.

Stage 2 surface: just enough to identify the agent, the interface, the target
SSID, the webcam, and where to bind the ZMQ sockets. CSI-specific knobs come
in Stage 3.
"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from shared.project import project_paths


@dataclass
class AgentConfig:
    project_dir: Path
    agent_id: str
    interface: str = "wlan1"
    target_ssid: str = ""
    target_bssid: str = ""
    # Window cadence; CSI capture in Stage 3 prefers ~2 s windows
    window_seconds: float = 2.0
    # Webcam
    webcam_enabled: bool = True
    webcam_index_hint: int = 0
    webcam_fps: float = 5.0
    webcam_jpeg_quality: int = 75
    webcam_max_width: int = 640
    webcam_max_height: int = 480
    # CSI (Stage 3 fills these in)
    csi_enabled: bool = False
    csi_channel: int = 6
    csi_bandwidth_mhz: int = 20
    csi_recv_port: int = 9300
    # Transport
    publish_bind: str = "tcp://0.0.0.0:5555"   # windows + health
    frames_bind: str = "tcp://0.0.0.0:5556"    # frames
    control_connect: Optional[str] = None      # REQ socket -> Mac REP (set when Mac is reachable)

    # Computed
    paths: dict = field(default_factory=dict)

    @classmethod
    def from_project(
        cls,
        project_dir: Path | str,
        *,
        agent_id: Optional[str] = None,
        overrides: Optional[dict] = None,
    ) -> "AgentConfig":
        p = Path(project_dir)
        paths = project_paths(p)
        cfg_path = paths["project_config"]
        cfg_data: dict = {}
        if cfg_path.exists():
            try:
                cfg_data = json.loads(cfg_path.read_text())
            except Exception:
                cfg_data = {}
        ac = cls(
            project_dir=p,
            agent_id=agent_id or cfg_data.get("agent_id") or socket.gethostname(),
            interface=cfg_data.get("default_interface", "wlan1"),
            target_ssid=cfg_data.get("target_ssid", ""),
            target_bssid=cfg_data.get("target_bssid", ""),
            paths=paths,
        )
        # Optional v2 sub-block in project_config.json
        v2 = cfg_data.get("v2_agent", {})
        for k, v in v2.items():
            if hasattr(ac, k):
                setattr(ac, k, v)
        if overrides:
            for k, v in overrides.items():
                if hasattr(ac, k) and v is not None:
                    setattr(ac, k, v)
        return ac
