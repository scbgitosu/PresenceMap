from __future__ import annotations

import datetime
from typing import List, Optional, Tuple

from shared.models import RoomLabel


def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-cast algorithm. Returns True if (x, y) is inside polygon."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    px, py = polygon[-1]
    for qx, qy in polygon:
        if ((qy > y) != (py > y)) and (x < (px - qx) * (y - qy) / (py - qy) + qx):
            inside = not inside
        px, py = qx, qy
    return inside


def infer_room(x: float, y: float, rooms: List[RoomLabel]) -> Optional[RoomLabel]:
    """Returns first RoomLabel whose polygon contains (x, y), or None."""
    for room in rooms:
        if room.polygon and point_in_polygon(x, y, room.polygon):
            return room
    return None


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string with millisecond precision."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def px_to_ft(distance_px: float, metadata: dict) -> float:
    """Convert image-pixel distance to feet using floorplan metadata scale."""
    scale = metadata.get("scale_pixels_per_foot")
    if not scale:
        raise ValueError(
            "Missing floorplan scale. Re-run floorplan_import.py and measure a known wall length."
        )
    return distance_px / float(scale)
