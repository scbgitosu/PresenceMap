"""
Generate Wi-Fi RSSI heatmaps from survey measurements.

Usage:
    python3 mac_analysis/heatmap_generator.py \
        --project survey_projects/apartment_test \
        --session baseline_current_router \
        --output-dir output/heatmaps
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image
from scipy.interpolate import griddata

from shared.utils import project_paths


# RSSI threshold colormap (discrete)
RSSI_BOUNDS = [-100, -75, -70, -67, -60, -50, 0]
RSSI_COLORS = ["#cc0000", "#ff4500", "#ff8c00", "#ffd700", "#90ee00", "#00c800"]
RSSI_LABELS = ["< −75 (Poor)", "−75 to −70", "−70 to −67 (Fair)", "−67 to −60", "−60 to −50", "≥ −50 (Excellent)"]

CMAP = mcolors.ListedColormap(RSSI_COLORS)
NORM = mcolors.BoundaryNorm(RSSI_BOUNDS, CMAP.N)


def _load_measurements(summary_csv: Path, session_id: Optional[str] = None) -> List[dict]:
    from shared.survey_metrics import load_measurements

    if not summary_csv.exists():
        raise FileNotFoundError(f"No measurements found at {summary_csv}")
    rows = load_measurements(summary_csv, session_id=session_id)
    if not rows:
        raise FileNotFoundError(f"No measurements found at {summary_csv}")
    return rows


def _load_rooms(rooms_json: Path) -> list:
    if not rooms_json.exists():
        return []
    with open(rooms_json, encoding="utf-8") as f:
        return json.load(f)


def _load_routers(router_json: Path) -> list:
    if not router_json.exists():
        return []
    with open(router_json, encoding="utf-8") as f:
        return json.load(f)


def _build_room_mask(rooms: list, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Return boolean mask where True = inside at least one room polygon."""
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    mask = np.zeros(len(points), dtype=bool)
    for room in rooms:
        poly = room.get("polygon", [])
        if len(poly) >= 3:
            mpl_path = MplPath(poly)
            mask |= mpl_path.contains_points(points)
    return mask.reshape(grid_x.shape)


def _draw_floorplan_base(ax, img_arr: np.ndarray):
    ax.imshow(img_arr, origin="upper", zorder=0)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_room_outlines(ax, rooms: list):
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    polys = []
    for room in rooms:
        poly = room.get("polygon", [])
        if len(poly) >= 3:
            patch = MplPolygon(poly, closed=True)
            polys.append(patch)
            lx = room.get("label_x")
            ly = room.get("label_y")
            if lx is not None and ly is not None:
                ax.text(lx, ly, room.get("room_name", ""), fontsize=8,
                        ha="center", va="center", color="#3333aa", alpha=0.8)
    if polys:
        pc = PatchCollection(polys, facecolor="none",
                             edgecolor="#6464c8", linewidth=1.2, alpha=0.6)
        ax.add_collection(pc)


def _draw_router_markers(ax, routers: list, highlight_position_id: Optional[str] = None):
    from mac_analysis.floorplan_viz import draw_trial_routers

    draw_trial_routers(ax, routers, highlight_position_id=highlight_position_id)


def generate_survey_points(
    measurements: List[dict],
    img_arr: np.ndarray,
    rooms: list,
    routers: list,
    output_path: Path,
    highlight_position_id: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(12, 9))
    _draw_floorplan_base(ax, img_arr)
    _draw_room_outlines(ax, rooms)

    xs = [m["x_px"] for m in measurements if m["rssi_avg_dbm"] is not None]
    ys = [m["y_px"] for m in measurements if m["rssi_avg_dbm"] is not None]
    vs = [m["rssi_avg_dbm"] for m in measurements if m["rssi_avg_dbm"] is not None]

    if xs:
        sc = ax.scatter(xs, ys, c=vs, cmap=CMAP, norm=NORM, s=120,
                        edgecolors="black", linewidths=0.8, zorder=15)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("RSSI (dBm)", fontsize=10)

    _draw_router_markers(ax, routers, highlight_position_id=highlight_position_id)
    ax.set_title("Survey measurement points", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_heatmap(
    measurements: List[dict],
    img_arr: np.ndarray,
    rooms: list,
    routers: list,
    output_path: Path,
    downsample: int = 4,
    highlight_position_id: Optional[str] = None,
):
    h, w = img_arr.shape[:2]
    gw, gh = w // downsample, h // downsample

    xs = np.array([m["x_px"] for m in measurements if m["rssi_avg_dbm"] is not None])
    ys = np.array([m["y_px"] for m in measurements if m["rssi_avg_dbm"] is not None])
    vs = np.array([m["rssi_avg_dbm"] for m in measurements if m["rssi_avg_dbm"] is not None])

    if len(xs) < 3:
        print("Not enough valid points to interpolate heatmap (need ≥ 3). Skipping.")
        return

    grid_xi, grid_yi = np.meshgrid(
        np.linspace(0, w, gw),
        np.linspace(0, h, gh),
    )

    grid_z = griddata((xs, ys), vs, (grid_xi, grid_yi), method="linear")
    # Fill NaN edges with nearest-neighbor
    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z_nearest = griddata((xs, ys), vs, (grid_xi, grid_yi), method="nearest")
        grid_z[nan_mask] = grid_z_nearest[nan_mask]

    # Mask outside room polygons
    if rooms:
        room_mask = _build_room_mask(rooms, grid_xi, grid_yi)
        grid_z_masked = np.where(room_mask, grid_z, np.nan)
    else:
        grid_z_masked = grid_z

    fig, ax = plt.subplots(figsize=(12, 9))
    _draw_floorplan_base(ax, img_arr)
    _draw_room_outlines(ax, rooms)

    im = ax.imshow(
        grid_z_masked,
        origin="upper",
        extent=[0, w, h, 0],
        cmap=CMAP,
        norm=NORM,
        alpha=0.55,
        zorder=5,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("RSSI (dBm)", fontsize=10)

    _draw_router_markers(ax, routers, highlight_position_id=highlight_position_id)
    ax.set_title("Wi-Fi RSSI Heatmap", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_weak_zones(
    measurements: List[dict],
    img_arr: np.ndarray,
    rooms: list,
    routers: list,
    output_path: Path,
    threshold_dbm: float = -70.0,
    downsample: int = 4,
    highlight_position_id: Optional[str] = None,
):
    h, w = img_arr.shape[:2]
    gw, gh = w // downsample, h // downsample

    xs = np.array([m["x_px"] for m in measurements if m["rssi_avg_dbm"] is not None])
    ys = np.array([m["y_px"] for m in measurements if m["rssi_avg_dbm"] is not None])
    vs = np.array([m["rssi_avg_dbm"] for m in measurements if m["rssi_avg_dbm"] is not None])

    if len(xs) < 3:
        print("Not enough valid points for weak-zones layer. Skipping.")
        return

    grid_xi, grid_yi = np.meshgrid(
        np.linspace(0, w, gw),
        np.linspace(0, h, gh),
    )
    grid_z = griddata((xs, ys), vs, (grid_xi, grid_yi), method="linear")
    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z_nearest = griddata((xs, ys), vs, (grid_xi, grid_yi), method="nearest")
        grid_z[nan_mask] = grid_z_nearest[nan_mask]

    # Only show pixels below threshold
    weak_mask = grid_z < threshold_dbm
    if rooms:
        room_mask = _build_room_mask(rooms, grid_xi, grid_yi)
        weak_mask = weak_mask & room_mask

    grid_weak = np.where(weak_mask, grid_z, np.nan)

    weak_cmap = mcolors.ListedColormap(["#cc0000", "#ff4500", "#ff8c00"])
    weak_norm = mcolors.BoundaryNorm([-100, -75, -70, threshold_dbm], weak_cmap.N)

    fig, ax = plt.subplots(figsize=(12, 9))
    _draw_floorplan_base(ax, img_arr)
    _draw_room_outlines(ax, rooms)

    im = ax.imshow(
        grid_weak,
        origin="upper",
        extent=[0, w, h, 0],
        cmap=weak_cmap,
        norm=weak_norm,
        alpha=0.6,
        zorder=5,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label(f"RSSI < {threshold_dbm} dBm (Weak zones)", fontsize=10)

    _draw_router_markers(ax, routers, highlight_position_id=highlight_position_id)
    ax.set_title(f"Weak signal zones (below {threshold_dbm} dBm)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_stats(measurements: List[dict], session_id: str):
    valid = [m for m in measurements if m["rssi_avg_dbm"] is not None]
    if not valid:
        print("No valid measurements.")
        return
    stats = measurement_stats(measurements)
    print(f"\n=== Stats for session '{session_id}' ===")
    print(f"  Total valid points : {stats['valid_count']}")
    print(f"  Average RSSI       : {stats['rssi_avg_dbm']:.1f} dBm")
    print(f"  Worst RSSI         : {stats['rssi_min_dbm']:.1f} dBm")
    print(f"  Points < -67 dBm   : {stats['count_below_67']} ({stats['pct_below_67']:.0f}%)")
    print(f"  Points < -70 dBm   : {stats['count_below_70']} ({stats['pct_below_70']:.0f}%)")
    print(f"  Points < -75 dBm   : {stats['count_below_75']} ({stats['pct_below_75']:.0f}%)")


def measurement_stats(measurements: List[dict]) -> dict:
    valid = [m for m in measurements if m["rssi_avg_dbm"] is not None]
    if not valid:
        return {
            "valid_count": 0,
            "rssi_avg_dbm": None,
            "rssi_min_dbm": None,
            "count_below_67": 0,
            "count_below_70": 0,
            "count_below_75": 0,
            "pct_below_67": 0.0,
            "pct_below_70": 0.0,
            "pct_below_75": 0.0,
        }
    rssi_vals = [m["rssi_avg_dbm"] for m in valid]
    avg = sum(rssi_vals) / len(rssi_vals)
    worst = min(rssi_vals)
    count = len(rssi_vals)
    c67 = sum(1 for v in rssi_vals if v < -67)
    c70 = sum(1 for v in rssi_vals if v < -70)
    c75 = sum(1 for v in rssi_vals if v < -75)
    return {
        "valid_count": count,
        "rssi_avg_dbm": avg,
        "rssi_min_dbm": worst,
        "count_below_67": c67,
        "count_below_70": c70,
        "count_below_75": c75,
        "pct_below_67": 100 * c67 / count,
        "pct_below_70": 100 * c70 / count,
        "pct_below_75": 100 * c75 / count,
    }


def run_heatmap_generation(
    project_dir: Path,
    session_id: str,
    output_dir: Path,
    weak_threshold: float = -70.0,
) -> dict:
    paths = project_paths(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dir = paths["survey_sessions_dir"] / session_id
    summary_csv = session_dir / "measurements_summary.csv"

    measurements = _load_measurements(summary_csv, session_id=session_id)
    rooms = _load_rooms(paths["rooms_json"])
    routers = _load_routers(paths["router_positions_json"])

    if not measurements:
        raise ValueError("No measurements found for that session.")

    img = Image.open(paths["floorplan_png"]).convert("RGB")
    img_arr = np.array(img)

    outputs = {
        "points": output_dir / f"{session_id}_points.png",
        "heatmap": output_dir / f"{session_id}_heatmap.png",
        "weak_zones": output_dir / f"{session_id}_weak_zones.png",
    }
    highlight = None
    if measurements:
        highlight = measurements[0].get("router_position_id") or None

    generate_survey_points(
        measurements, img_arr, rooms, routers, outputs["points"],
        highlight_position_id=highlight,
    )
    generate_heatmap(
        measurements, img_arr, rooms, routers, outputs["heatmap"],
        highlight_position_id=highlight,
    )
    generate_weak_zones(
        measurements,
        img_arr,
        rooms,
        routers,
        outputs["weak_zones"],
        threshold_dbm=weak_threshold,
        highlight_position_id=highlight,
    )
    return {
        "session_id": session_id,
        "outputs": outputs,
        "stats": measurement_stats(measurements),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Wi-Fi heatmaps")
    parser.add_argument("--project", required=True, help="Path to project directory")
    parser.add_argument("--session", required=True, help="Session name (session_id)")
    parser.add_argument("--output-dir", default="output/heatmaps", help="Output directory")
    parser.add_argument("--weak-threshold", type=float, default=-70.0,
                        help="dBm threshold for weak-zones layer")
    args = parser.parse_args()

    try:
        result = run_heatmap_generation(
            Path(args.project),
            args.session,
            Path(args.output_dir),
            weak_threshold=args.weak_threshold,
        )
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    stats = result["stats"]
    print(f"\n=== Stats for session '{args.session}' ===")
    if stats["valid_count"]:
        print(f"  Total valid points : {stats['valid_count']}")
        print(f"  Average RSSI       : {stats['rssi_avg_dbm']:.1f} dBm")
        print(f"  Worst RSSI         : {stats['rssi_min_dbm']:.1f} dBm")
        print(f"  Points < -67 dBm   : {stats['count_below_67']} ({stats['pct_below_67']:.0f}%)")
        print(f"  Points < -70 dBm   : {stats['count_below_70']} ({stats['pct_below_70']:.0f}%)")
        print(f"  Points < -75 dBm   : {stats['count_below_75']} ({stats['pct_below_75']:.0f}%)")
    else:
        print("No valid measurements.")


if __name__ == "__main__":
    main()
