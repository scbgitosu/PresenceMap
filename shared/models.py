from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Project / floorplan dataclasses (used by mac_app/floorplan/ and v2 dashboard)
# --------------------------------------------------------------------------- #


@dataclass
class RoomLabel:
    room_id: str
    room_name: str
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    label_x: Optional[float] = None
    label_y: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "room_name": self.room_name,
            "polygon": self.polygon,
            "label_x": self.label_x,
            "label_y": self.label_y,
        }


@dataclass
class RouterPosition:
    router_position_id: str
    name: str
    x_px: float
    y_px: float
    height_ft: float = 4.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProjectConfig:
    project_name: str
    target_ssid: str
    target_bssid: str
    default_interface: str
    units: str
    collection_mode: str
    scan_backend: str = "iw"
    paths: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# v2 runtime dataclasses (filled in across stages 2–5)
# --------------------------------------------------------------------------- #


@dataclass
class AgentMeta:
    """Identifies the HP sensor agent that produced a message."""

    agent_id: str
    interface: str           # e.g. "wlan1"
    backend: str             # e.g. "atheros_csi" | "iw_scan"
    schema_version: int = 2


@dataclass
class CSIWindow:
    """Per-window CSI feature aggregate from one HP agent."""

    window_id: str
    session_id: str
    ts_start: str
    ts_end: str
    features: List[float] = field(default_factory=list)  # ~176 floats
    frames: int = 0
    loss_ratio: float = 0.0
    amp_matrix_f16: Optional[bytes] = None  # training-only blob


@dataclass
class RFWindow:
    """Per-window RSSI / iw-scan aggregate, sibling to CSIWindow."""

    target_rssi_avg_dbm: Optional[float] = None
    target_rssi_std_db: Optional[float] = None
    snr_db: Optional[float] = None
    noise_dbm: Optional[float] = None
    visible_bssid_count: int = 0
    neighbor_rssi_sum_dbm: Optional[float] = None
    channel_utilization_proxy: Optional[float] = None
    target_seen: bool = False


@dataclass
class GTLabel:
    """Ground-truth occupancy label for a single RF window."""

    window_id: str
    session_id: str
    ts_start: str
    ts_end: str
    occupancy: str             # "vacant" | "occupied" | "unknown"
    person_count: int = 0
    label_confidence: float = 0.0
    label_source: str = "yolo"  # "yolo" | "manual" | "missing"
    yolo_frame_refs: List[str] = field(default_factory=list)
    yolo_model_id: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
