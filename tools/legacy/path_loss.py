"""Path-loss fitting and prediction helpers for Wi-Fi placement optimization."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from shared.utils import point_in_polygon


@dataclass
class PathLossModel:
    p0_dbm: float
    exponent_n: float
    cross_room_penalty_db: float
    rmse_db: float
    sample_count: int


def infer_room_id(x_px: float, y_px: float, rooms: list) -> str:
    for room in rooms:
        poly = room.get("polygon", [])
        if len(poly) >= 3 and point_in_polygon(x_px, y_px, poly):
            return room.get("room_id", "")
    return ""


def distance_ft(
    ax_px: float,
    ay_px: float,
    ah_ft: float,
    bx_px: float,
    by_px: float,
    bh_ft: float,
    scale_pixels_per_foot: float,
) -> float:
    dx_ft = (ax_px - bx_px) / scale_pixels_per_foot
    dy_ft = (ay_px - by_px) / scale_pixels_per_foot
    dz_ft = ah_ft - bh_ft
    return max(1.0, math.sqrt(dx_ft * dx_ft + dy_ft * dy_ft + dz_ft * dz_ft))


def build_training_rows(
    measurements_by_session: Dict[str, List[dict]],
    routers_by_id: Dict[str, dict],
    rooms: list,
    scale_pixels_per_foot: float,
) -> list:
    rows = []
    for session_id, measurements in measurements_by_session.items():
        for m in measurements:
            rssi = m.get("rssi_avg_dbm")
            router_id = m.get("router_position_id", "")
            router = routers_by_id.get(router_id)
            if rssi is None or router is None:
                continue
            ap_room = infer_room_id(router["x_px"], router["y_px"], rooms)
            rx_room = m.get("room_id") or infer_room_id(m["x_px"], m["y_px"], rooms)
            dist = distance_ft(
                float(router["x_px"]),
                float(router["y_px"]),
                float(router.get("height_ft", 4.0)),
                float(m["x_px"]),
                float(m["y_px"]),
                float(m.get("height_ft") or 4.0),
                scale_pixels_per_foot,
            )
            rows.append({
                "session_id": session_id,
                "router_position_id": router_id,
                "distance_ft": dist,
                "cross_room": 1.0 if ap_room and rx_room and ap_room != rx_room else 0.0,
                "rssi_dbm": float(rssi),
            })
    return rows


def fit_path_loss(rows: list) -> PathLossModel:
    if len(rows) < 3:
        raise ValueError("Need at least three valid measurements to fit a path-loss model.")

    x = []
    y = []
    for row in rows:
        x.append([1.0, -10.0 * math.log10(row["distance_ft"]), -row["cross_room"]])
        y.append(row["rssi_dbm"])
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    coeffs, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
    p0, exponent_n, cross_room_penalty = coeffs
    exponent_n = float(max(1.0, min(6.0, exponent_n)))
    cross_room_penalty = float(max(0.0, cross_room_penalty))

    preds = predict_rows(rows, PathLossModel(float(p0), exponent_n, cross_room_penalty, 0.0, len(rows)))
    rmse = float(np.sqrt(np.mean([(pred - row["rssi_dbm"]) ** 2 for pred, row in zip(preds, rows)])))
    return PathLossModel(float(p0), exponent_n, cross_room_penalty, rmse, len(rows))


def predict_rssi(
    model: PathLossModel,
    distance_ft_value: float,
    *,
    cross_room: bool = False,
) -> float:
    rssi = model.p0_dbm - 10.0 * model.exponent_n * math.log10(max(1.0, distance_ft_value))
    if cross_room:
        rssi -= model.cross_room_penalty_db
    return rssi


def predict_rows(rows: list, model: PathLossModel) -> list:
    return [
        predict_rssi(model, row["distance_ft"], cross_room=bool(row["cross_room"]))
        for row in rows
    ]


def leave_one_session_out_rmse(rows: list) -> Optional[float]:
    sessions = sorted({row["session_id"] for row in rows})
    if len(sessions) < 2:
        return None
    errors = []
    for session in sessions:
        train = [row for row in rows if row["session_id"] != session]
        test = [row for row in rows if row["session_id"] == session]
        if len(train) < 3 or not test:
            continue
        model = fit_path_loss(train)
        for row in test:
            pred = predict_rssi(model, row["distance_ft"], cross_room=bool(row["cross_room"]))
            errors.append((pred - row["rssi_dbm"]) ** 2)
    if not errors:
        return None
    return float(np.sqrt(np.mean(errors)))
