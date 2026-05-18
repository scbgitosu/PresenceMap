"""
Match survey clicks across sessions so router trials compare the same locations.

Usage:
    python3 mac_analysis/walk_matcher.py \
        --project survey_projects/apartment_test \
        --sessions trial_pos1,trial_pos2,trial_pos3 \
        --output output/comparison/walk_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.survey_metrics import discover_sessions, load_measurements
from shared.utils import project_paths


def _load_waypoints(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return sorted(json.load(f), key=lambda w: int(w.get("order", 0)))


def _distance(a: dict, b: dict) -> float:
    dx = float(a["x_px"]) - float(b["x_px"])
    dy = float(a["y_px"]) - float(b["y_px"])
    return math.hypot(dx, dy)


def _match_by_waypoint(
    session_rows: Dict[str, List[dict]],
    waypoints: list,
) -> Dict[str, Dict[str, dict]]:
    matched: Dict[str, Dict[str, dict]] = {}
    for waypoint in waypoints:
        waypoint_id = waypoint.get("waypoint_id", "")
        if not waypoint_id:
            continue
        matched[waypoint_id] = {}
        for session_id, rows in session_rows.items():
            row = next((r for r in rows if r.get("waypoint_id") == waypoint_id), None)
            if row is None:
                # Fall back to order if older rows do not include waypoint_id.
                order = int(waypoint.get("order", 0))
                if 1 <= order <= len(rows):
                    row = rows[order - 1]
            if row is not None:
                matched[waypoint_id][session_id] = row
    return matched


def _match_by_nearest(
    session_rows: Dict[str, List[dict]],
    reference_session: str,
    tolerance_px: float,
) -> Dict[str, Dict[str, dict]]:
    reference_rows = session_rows[reference_session]
    matched: Dict[str, Dict[str, dict]] = {}
    used: Dict[str, set] = {sid: set() for sid in session_rows if sid != reference_session}

    for i, ref in enumerate(reference_rows, start=1):
        waypoint_id = f"match_{i:02d}"
        matched[waypoint_id] = {reference_session: ref}
        for session_id, rows in session_rows.items():
            if session_id == reference_session:
                continue
            best_idx = None
            best_dist = None
            for idx, row in enumerate(rows):
                if idx in used[session_id]:
                    continue
                dist = _distance(ref, row)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist is not None and best_dist <= tolerance_px:
                used[session_id].add(best_idx)
                matched[waypoint_id][session_id] = rows[best_idx]
    return matched


def build_walk_pairs(
    project_dir: Path,
    session_ids: List[str],
    *,
    tolerance_px: float = 40.0,
) -> list:
    paths = project_paths(project_dir)
    session_rows = {
        sid: load_measurements(paths["survey_sessions_dir"] / sid / "measurements_summary.csv", sid)
        for sid in session_ids
    }
    session_rows = {sid: rows for sid, rows in session_rows.items() if rows}
    if len(session_rows) < 2:
        return []

    waypoints = _load_waypoints(paths["walk_waypoints_json"])
    if waypoints:
        matched = _match_by_waypoint(session_rows, waypoints)
    else:
        matched = _match_by_nearest(session_rows, session_ids[0], tolerance_px)

    pairs = []
    for waypoint_id, rows_by_session in matched.items():
        present_sessions = [sid for sid in session_ids if sid in rows_by_session]
        for i, session_a in enumerate(present_sessions):
            for session_b in present_sessions[i + 1:]:
                a = rows_by_session[session_a]
                b = rows_by_session[session_b]
                rssi_a = a.get("rssi_avg_dbm")
                rssi_b = b.get("rssi_avg_dbm")
                pairs.append({
                    "waypoint_id": waypoint_id,
                    "session_a": session_a,
                    "click_id_a": a.get("click_id", ""),
                    "session_b": session_b,
                    "click_id_b": b.get("click_id", ""),
                    "x_px": (a["x_px"] + b["x_px"]) / 2,
                    "y_px": (a["y_px"] + b["y_px"]) / 2,
                    "delta_rssi_dbm": (rssi_b - rssi_a) if rssi_a is not None and rssi_b is not None else "",
                    "rssi_a_dbm": rssi_a if rssi_a is not None else "",
                    "rssi_b_dbm": rssi_b if rssi_b is not None else "",
                    "distance_px": _distance(a, b),
                })
    return pairs


def write_walk_pairs(pairs: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "waypoint_id",
        "session_a",
        "click_id_a",
        "session_b",
        "click_id_b",
        "x_px",
        "y_px",
        "delta_rssi_dbm",
        "rssi_a_dbm",
        "rssi_b_dbm",
        "distance_px",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Match walk points across survey sessions")
    parser.add_argument("--project", required=True, help="Path to project directory")
    parser.add_argument("--sessions", default="", help="Comma-separated session IDs")
    parser.add_argument("--output", default="output/comparison/walk_pairs.csv")
    parser.add_argument("--tolerance-px", type=float, default=40.0)
    args = parser.parse_args()

    project_dir = Path(args.project)
    paths = project_paths(project_dir)
    if args.sessions.strip():
        session_ids = [s.strip() for s in args.sessions.split(",") if s.strip()]
    else:
        session_ids = discover_sessions(paths["survey_sessions_dir"])
    pairs = build_walk_pairs(project_dir, session_ids, tolerance_px=args.tolerance_px)
    if not pairs:
        print("No matched walk pairs found.")
        return
    write_walk_pairs(pairs, Path(args.output))


if __name__ == "__main__":
    main()
