"""
Survey measurement metrics for point-level stats and interpolated coverage grids.
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.interpolate import griddata

from shared.utils import point_in_polygon, project_paths


DEFAULT_GOOD_DBM = -67.0
DEFAULT_WEAK_DBM = -70.0
DEFAULT_VARIANCE_DBM = 4.0  # rssi_std above this → "unstable" point


@dataclass
class PointMetrics:
    """Aggregated stats from measurement click points (not the interpolated grid)."""
    session_id: str
    router_position_id: str
    point_count: int = 0
    valid_count: int = 0
    rssi_mean: Optional[float] = None
    rssi_median: Optional[float] = None
    rssi_min: Optional[float] = None
    rssi_max: Optional[float] = None
    rssi_p10: Optional[float] = None
    rssi_p90: Optional[float] = None
    mean_rssi_std: Optional[float] = None
    pct_below_good: float = 0.0
    pct_below_weak: float = 0.0
    pct_below_poor: float = 0.0  # < -75 dBm
    pct_unstable: float = 0.0  # rssi_std > threshold
    pct_missing_samples: float = 0.0
    total_missing_samples: int = 0
    snr_mean: Optional[float] = None
    snr_min: Optional[float] = None
    tx_bitrate_mean: Optional[float] = None
    rx_bitrate_mean: Optional[float] = None
    neighbor_count_same_channel_mean: Optional[float] = None
    neighbor_count_adjacent_mean: Optional[float] = None
    channel_utilization_proxy_mean: Optional[float] = None


@dataclass
class GridCoverageMetrics:
    """Stats from scipy griddata interpolation over the floorplan."""
    grid_width: int = 0
    grid_height: int = 0
    masked_cell_count: int = 0
    valid_cell_count: int = 0
    rssi_grid_mean: Optional[float] = None
    rssi_grid_min: Optional[float] = None
    rssi_grid_p10: Optional[float] = None
    pct_area_good: float = 0.0
    pct_area_weak: float = 0.0
    pct_area_poor: float = 0.0


@dataclass
class RoomMetrics:
    room_id: str
    room_name: str
    point_count: int
    rssi_mean: Optional[float]
    rssi_min: Optional[float]
    pct_below_good: float


@dataclass
class SessionMetrics:
    session_id: str
    router_position_id: str
    router_name: str = ""
    points: PointMetrics = field(default_factory=PointMetrics)
    grid: GridCoverageMetrics = field(default_factory=GridCoverageMetrics)
    rooms: List[RoomMetrics] = field(default_factory=list)
    composite_score: float = 0.0
    rank: int = 0


def load_measurements(summary_csv: Path, session_id: Optional[str] = None) -> List[dict]:
    if not summary_csv.exists():
        return []
    # Always scope to the session folder name so stray rows from other trials are ignored.
    if session_id is None and summary_csv.parent.name not in (".", "survey_sessions"):
        session_id = summary_csv.parent.name
    rows = []
    skipped_foreign = 0
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if session_id and row.get("session_id") != session_id:
                skipped_foreign += 1
                continue
            try:
                row["x_px"] = float(row["x_px"])
                row["y_px"] = float(row["y_px"])
                for key in (
                    "rssi_avg_dbm",
                    "rssi_min_dbm",
                    "rssi_max_dbm",
                    "rssi_std_db",
                    "noise_avg_dbm",
                    "snr_avg_db",
                    "snr_min_db",
                    "tx_bitrate_avg_mbps",
                    "rx_bitrate_avg_mbps",
                    "neighbor_count_same_channel",
                    "neighbor_count_adjacent",
                    "neighbor_rssi_sum_dbm",
                    "channel_utilization_proxy",
                    "height_ft",
                    "sample_count",
                    "missing_sample_count",
                ):
                    raw = row.get(key)
                    row[key] = float(raw) if raw not in (None, "") else None
            except ValueError:
                continue
            rows.append(row)
    if skipped_foreign:
        import logging

        logging.getLogger(__name__).warning(
            "Skipped %d summary row(s) in %s with session_id != %s",
            skipped_foreign,
            summary_csv,
            session_id,
        )
    return rows


def load_rooms(rooms_json: Path) -> list:
    if not rooms_json.exists():
        return []
    with open(rooms_json, encoding="utf-8") as f:
        return json.load(f)


def load_routers(router_json: Path) -> list:
    if not router_json.exists():
        return []
    with open(router_json, encoding="utf-8") as f:
        return json.load(f)


def router_name_map(routers: list) -> Dict[str, str]:
    return {r["router_position_id"]: r.get("name", r["router_position_id"]) for r in routers}


def discover_sessions(survey_sessions_dir: Path) -> List[str]:
    if not survey_sessions_dir.is_dir():
        return []
    found = []
    for child in sorted(survey_sessions_dir.iterdir()):
        if child.is_dir() and (child / "measurements_summary.csv").exists():
            found.append(child.name)
    return found


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def compute_point_metrics(
    measurements: List[dict],
    session_id: str,
    router_position_id: str,
    *,
    good_dbm: float = DEFAULT_GOOD_DBM,
    weak_dbm: float = DEFAULT_WEAK_DBM,
    variance_dbm: float = DEFAULT_VARIANCE_DBM,
) -> PointMetrics:
    pm = PointMetrics(session_id=session_id, router_position_id=router_position_id)
    pm.point_count = len(measurements)
    valid = [m for m in measurements if m.get("rssi_avg_dbm") is not None]
    pm.valid_count = len(valid)
    if not valid:
        return pm

    avgs = [m["rssi_avg_dbm"] for m in valid]
    pm.rssi_mean = statistics.mean(avgs)
    pm.rssi_median = statistics.median(avgs)
    pm.rssi_min = min(avgs)
    pm.rssi_max = max(avgs)
    pm.rssi_p10 = _percentile(avgs, 10)
    pm.rssi_p90 = _percentile(avgs, 90)

    stds = [m["rssi_std_db"] for m in valid if m.get("rssi_std_db") is not None]
    if stds:
        pm.mean_rssi_std = statistics.mean(stds)
        pm.pct_unstable = 100.0 * sum(1 for s in stds if s > variance_dbm) / len(stds)

    pm.pct_below_good = 100.0 * sum(1 for v in avgs if v < good_dbm) / len(avgs)
    pm.pct_below_weak = 100.0 * sum(1 for v in avgs if v < weak_dbm) / len(avgs)
    pm.pct_below_poor = 100.0 * sum(1 for v in avgs if v < -75) / len(avgs)

    missing_flags = [m for m in measurements if (m.get("missing_sample_count") or 0) > 0]
    pm.pct_missing_samples = 100.0 * len(missing_flags) / pm.point_count if pm.point_count else 0.0
    pm.total_missing_samples = int(
        sum(m.get("missing_sample_count") or 0 for m in measurements)
    )
    snrs = [m["snr_avg_db"] for m in valid if m.get("snr_avg_db") is not None]
    if snrs:
        pm.snr_mean = statistics.mean(snrs)
        pm.snr_min = min(snrs)
    tx_rates = [m["tx_bitrate_avg_mbps"] for m in valid if m.get("tx_bitrate_avg_mbps") is not None]
    rx_rates = [m["rx_bitrate_avg_mbps"] for m in valid if m.get("rx_bitrate_avg_mbps") is not None]
    if tx_rates:
        pm.tx_bitrate_mean = statistics.mean(tx_rates)
    if rx_rates:
        pm.rx_bitrate_mean = statistics.mean(rx_rates)
    same_neighbors = [
        m["neighbor_count_same_channel"]
        for m in valid
        if m.get("neighbor_count_same_channel") is not None
    ]
    adjacent_neighbors = [
        m["neighbor_count_adjacent"]
        for m in valid
        if m.get("neighbor_count_adjacent") is not None
    ]
    channel_proxy = [
        m["channel_utilization_proxy"]
        for m in valid
        if m.get("channel_utilization_proxy") is not None
    ]
    if same_neighbors:
        pm.neighbor_count_same_channel_mean = statistics.mean(same_neighbors)
    if adjacent_neighbors:
        pm.neighbor_count_adjacent_mean = statistics.mean(adjacent_neighbors)
    if channel_proxy:
        pm.channel_utilization_proxy_mean = statistics.mean(channel_proxy)
    return pm


def build_room_mask(rooms: list, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    mask = np.zeros(len(points), dtype=bool)
    for room in rooms:
        poly = room.get("polygon", [])
        if len(poly) >= 3:
            mask |= MplPath(poly).contains_points(points)
    return mask.reshape(grid_x.shape)


def interpolate_rssi_grid(
    measurements: List[dict],
    image_width: int,
    image_height: int,
    *,
    downsample: int = 4,
    rooms: Optional[list] = None,
) -> Optional[np.ndarray]:
    xs = np.array([m["x_px"] for m in measurements if m.get("rssi_avg_dbm") is not None])
    ys = np.array([m["y_px"] for m in measurements if m.get("rssi_avg_dbm") is not None])
    vs = np.array([m["rssi_avg_dbm"] for m in measurements if m.get("rssi_avg_dbm") is not None])

    if len(xs) < 3:
        return None

    gw, gh = image_width // downsample, image_height // downsample
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, image_width, gw),
        np.linspace(0, image_height, gh),
    )
    grid_z = griddata((xs, ys), vs, (grid_x, grid_y), method="linear")
    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z_nearest = griddata((xs, ys), vs, (grid_x, grid_y), method="nearest")
        grid_z[nan_mask] = grid_z_nearest[nan_mask]

    if rooms:
        room_mask = build_room_mask(rooms, grid_x, grid_y)
        grid_z = np.where(room_mask, grid_z, np.nan)
    return grid_z


def compute_grid_metrics(
    grid_z: Optional[np.ndarray],
    *,
    good_dbm: float = DEFAULT_GOOD_DBM,
    weak_dbm: float = DEFAULT_WEAK_DBM,
) -> GridCoverageMetrics:
    gm = GridCoverageMetrics()
    if grid_z is None:
        return gm

    gm.grid_height, gm.grid_width = grid_z.shape
    valid = grid_z[~np.isnan(grid_z)]
    gm.masked_cell_count = int(np.sum(~np.isnan(grid_z)))
    gm.valid_cell_count = len(valid)
    if len(valid) == 0:
        return gm

    gm.rssi_grid_mean = float(np.mean(valid))
    gm.rssi_grid_min = float(np.min(valid))
    gm.rssi_grid_p10 = float(np.percentile(valid, 10))
    gm.pct_area_good = 100.0 * float(np.sum(valid >= good_dbm)) / len(valid)
    gm.pct_area_weak = 100.0 * float(np.sum(valid < weak_dbm)) / len(valid)
    gm.pct_area_poor = 100.0 * float(np.sum(valid < -75)) / len(valid)
    return gm


def compute_room_metrics(
    measurements: List[dict],
    rooms: list,
    *,
    good_dbm: float = DEFAULT_GOOD_DBM,
) -> List[RoomMetrics]:
    if not rooms:
        return []

    by_room: Dict[str, List[float]] = {}
    names: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    for m in measurements:
        if m.get("rssi_avg_dbm") is None:
            continue
        x, y = m["x_px"], m["y_px"]
        matched_id = m.get("room_id") or ""
        matched_name = m.get("room_name") or ""
        if not matched_id:
            for room in rooms:
                poly = room.get("polygon", [])
                if len(poly) >= 3 and point_in_polygon(x, y, poly):
                    matched_id = room.get("room_id", "")
                    matched_name = room.get("room_name", matched_id)
                    break
        if not matched_id:
            matched_id = "_unassigned"
            matched_name = "Unassigned"
        by_room.setdefault(matched_id, []).append(m["rssi_avg_dbm"])
        names[matched_id] = matched_name
        counts[matched_id] = counts.get(matched_id, 0) + 1

    result = []
    for room_id, vals in sorted(by_room.items()):
        result.append(
            RoomMetrics(
                room_id=room_id,
                room_name=names.get(room_id, room_id),
                point_count=counts[room_id],
                rssi_mean=statistics.mean(vals),
                rssi_min=min(vals),
                pct_below_good=100.0 * sum(1 for v in vals if v < good_dbm) / len(vals),
            )
        )
    return result


def composite_score(
    points: PointMetrics,
    grid: GridCoverageMetrics,
    *,
    good_dbm: float = DEFAULT_GOOD_DBM,
) -> float:
    """
    Higher is better (0–100 scale). Weights emphasize widespread coverage and
    worst-case performance; stability and scan reliability are secondary.
    """
    if points.valid_count == 0:
        return 0.0

    # Grid coverage (0–40 pts) — primary for "widespread strength"
    if grid.valid_cell_count > 0:
        grid_part = 0.40 * grid.pct_area_good
    else:
        grid_part = 0.40 * (100.0 - points.pct_below_good)

    # Worst-case: 10th percentile of point RSSI mapped to 0–25 pts
    p10 = points.rssi_p10 if points.rssi_p10 is not None else points.rssi_min
    if p10 is not None:
        # -50 → 25, -75 → 0, linear between
        worst_part = 25.0 * max(0.0, min(1.0, (p10 + 75) / 25.0))
    else:
        worst_part = 0.0

    # Mean RSSI (0–20 pts): -50 → 20, -80 → 0
    mean_part = 0.0
    if points.rssi_mean is not None:
        mean_part = 20.0 * max(0.0, min(1.0, (points.rssi_mean + 80) / 30.0))

    # Stability (0–10 pts): lower std is better
    stab_part = 10.0
    if points.mean_rssi_std is not None:
        stab_part = max(0.0, 10.0 - points.mean_rssi_std * 2.0)

    # Reliability (0–5 pts): penalize missing subsamples
    rel_part = max(0.0, 5.0 - points.pct_missing_samples * 0.05)

    # RF health adjustment: reward SNR and penalize visible co-channel congestion.
    snr_part = 0.0
    if points.snr_mean is not None:
        snr_part = 8.0 * max(0.0, min(1.0, points.snr_mean / 35.0))
    interference_penalty = 0.0
    if points.channel_utilization_proxy_mean is not None:
        interference_penalty = min(8.0, points.channel_utilization_proxy_mean * 0.08)

    return round(
        grid_part + worst_part + mean_part + stab_part + rel_part + snr_part - interference_penalty,
        1,
    )


def analyze_session(
    session_id: str,
    summary_csv: Path,
    *,
    image_size: Optional[tuple[int, int]] = None,
    rooms: Optional[list] = None,
    routers_by_id: Optional[Dict[str, str]] = None,
    good_dbm: float = DEFAULT_GOOD_DBM,
    weak_dbm: float = DEFAULT_WEAK_DBM,
    downsample: int = 4,
) -> SessionMetrics:
    measurements = load_measurements(summary_csv, session_id=session_id)
    router_position_id = ""
    if measurements:
        router_position_id = measurements[0].get("router_position_id", "") or ""

    points = compute_point_metrics(
        measurements,
        session_id,
        router_position_id,
        good_dbm=good_dbm,
        weak_dbm=weak_dbm,
    )

    grid = GridCoverageMetrics()
    if image_size and len(measurements) >= 3:
        w, h = image_size
        grid_z = interpolate_rssi_grid(
            measurements, w, h, downsample=downsample, rooms=rooms
        )
        grid = compute_grid_metrics(grid_z, good_dbm=good_dbm, weak_dbm=weak_dbm)

    room_metrics = compute_room_metrics(measurements, rooms or [], good_dbm=good_dbm)

    sm = SessionMetrics(
        session_id=session_id,
        router_position_id=router_position_id,
        router_name=(routers_by_id or {}).get(router_position_id, router_position_id),
        points=points,
        grid=grid,
        rooms=room_metrics,
    )
    sm.composite_score = composite_score(points, grid, good_dbm=good_dbm)
    return sm


def analyze_project(
    project_dir: Path,
    session_ids: Optional[List[str]] = None,
    *,
    good_dbm: float = DEFAULT_GOOD_DBM,
    weak_dbm: float = DEFAULT_WEAK_DBM,
    downsample: int = 4,
) -> List[SessionMetrics]:
    paths = project_paths(project_dir)
    rooms = load_rooms(paths["rooms_json"])
    routers = load_routers(paths["router_positions_json"])
    names = router_name_map(routers)

    if session_ids is None:
        session_ids = discover_sessions(paths["survey_sessions_dir"])
    if not session_ids:
        return []

    image_size = None
    floorplan = paths["floorplan_png"]
    if floorplan.exists():
        from PIL import Image

        with Image.open(floorplan) as img:
            image_size = img.size  # (width, height)

    results = []
    for sid in session_ids:
        summary_csv = paths["survey_sessions_dir"] / sid / "measurements_summary.csv"
        results.append(
            analyze_session(
                sid,
                summary_csv,
                image_size=image_size,
                rooms=rooms,
                routers_by_id=names,
                good_dbm=good_dbm,
                weak_dbm=weak_dbm,
                downsample=downsample,
            )
        )

    ranked = sorted(results, key=lambda s: s.composite_score, reverse=True)
    for i, sm in enumerate(ranked, start=1):
        sm.rank = i
    rank_by_session = {sm.session_id: sm.rank for sm in ranked}
    for sm in results:
        sm.rank = rank_by_session[sm.session_id]
    return results
