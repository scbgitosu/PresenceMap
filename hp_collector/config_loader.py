"""Load and validate project configuration files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from shared.models import ProjectConfig, RoomLabel, RouterPosition
from shared.utils import project_paths


def _require(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name} — {hint}. Expected at: {path}"
        )


def _load_json_if_exists(path: Path, default):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_project(project_dir: Path, *, require_spatial: bool = True) -> tuple:
    """
    Load all project files. Returns (ProjectConfig, List[RoomLabel], List[RouterPosition], metadata).
    Raises FileNotFoundError with friendly messages for missing files.
    """
    paths = project_paths(Path(project_dir))

    _require(
        paths["project_config"],
        "create a PresenceMap project config from the dashboard or pass explicit --ssid/--interface values",
    )

    if require_spatial:
        _require(
            paths["floorplan_png"],
            "optional for PresenceMap, required for HeatMap survey mode; run floorplan_import.py on the Mac first",
        )
        _require(
            paths["rooms_json"],
            "optional for PresenceMap, required for HeatMap survey mode; run floorplan_labeler.py on the Mac first",
        )
        _require(
            paths["router_positions_json"],
            "optional for PresenceMap, required for HeatMap survey mode; run floorplan_labeler.py on the Mac first",
        )

    cfg_data = _load_json_if_exists(paths["project_config"], {})

    config = ProjectConfig(
        project_name=cfg_data.get("project_name", ""),
        target_ssid=cfg_data.get("target_ssid", ""),
        target_bssid=cfg_data.get("target_bssid", ""),
        default_interface=cfg_data.get("default_interface", ""),
        units=cfg_data.get("units", "feet"),
        collection_mode=cfg_data.get("collection_mode", "click_to_scan"),
        scan_backend=cfg_data.get("scan_backend", "iw"),
        paths=cfg_data.get("paths", {}),
    )

    rooms_data = _load_json_if_exists(paths["rooms_json"], [])

    rooms: List[RoomLabel] = []
    for r in rooms_data:
        rooms.append(RoomLabel(
            room_id=r.get("room_id", ""),
            room_name=r.get("room_name", ""),
            polygon=[tuple(p) for p in r.get("polygon", [])],
            label_x=r.get("label_x"),
            label_y=r.get("label_y"),
        ))

    router_data = _load_json_if_exists(paths["router_positions_json"], [])

    routers: List[RouterPosition] = []
    for rp in router_data:
        routers.append(RouterPosition(
            router_position_id=rp.get("router_position_id", ""),
            name=rp.get("name", ""),
            x_px=rp.get("x_px", 0),
            y_px=rp.get("y_px", 0),
            height_ft=rp.get("height_ft", 4.0),
            notes=rp.get("notes", ""),
        ))

    metadata = _load_json_if_exists(paths["floorplan_metadata"], {})

    return config, rooms, routers, metadata
