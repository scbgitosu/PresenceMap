"""Repair survey session CSVs (wrong session rows, missing waypoint_id, off-path clicks)."""
from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from shared.csv_schema import RAW_COLUMNS, SUMMARY_COLUMNS


@dataclass
class CleanupReport:
    session_id: str
    summary_before: int = 0
    summary_after: int = 0
    raw_before: int = 0
    raw_after: int = 0
    removed_foreign_session: int = 0
    removed_off_path: int = 0
    backfilled_waypoint_id: int = 0
    warnings: List[str] = field(default_factory=list)


def _load_waypoints(waypoints_json: Path) -> list:
    if not waypoints_json.exists():
        return []
    with open(waypoints_json, encoding="utf-8") as f:
        return sorted(json.load(f), key=lambda w: int(w.get("order", 0)))


def _nearest_waypoint(x: float, y: float, waypoints: list) -> tuple[Optional[dict], float]:
    best = None
    best_dist = float("inf")
    for wp in waypoints:
        dx = float(wp["x_px"]) - x
        dy = float(wp["y_px"]) - y
        dist = math.hypot(dx, dy)
        if dist < best_dist:
            best_dist = dist
            best = wp
    return best, best_dist


def clean_session_folder(
    session_dir: Path,
    *,
    waypoints_json: Optional[Path] = None,
    snap_tolerance_px: float = 40.0,
    drop_off_path: bool = True,
    backup: bool = True,
) -> CleanupReport:
    """
    Keep only rows for this session folder, optionally drop off-template clicks,
    and backfill empty waypoint_id when within snap tolerance.
    """
    session_dir = Path(session_dir)
    session_id = session_dir.name
    report = CleanupReport(session_id=session_id)

    summary_path = session_dir / "measurements_summary.csv"
    raw_path = session_dir / "measurements_raw.csv"
    if not summary_path.exists():
        report.warnings.append("No measurements_summary.csv")
        return report

    if backup:
        for path in (summary_path, raw_path):
            if path.exists():
                shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    with open(summary_path, newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    report.summary_before = len(summary_rows)

    raw_rows: List[dict] = []
    if raw_path.exists():
        with open(raw_path, newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
    report.raw_before = len(raw_rows)

    waypoints = _load_waypoints(waypoints_json) if waypoints_json else []

    kept_summary: List[dict] = []
    kept_click_ids: set[str] = set()

    for row in summary_rows:
        if row.get("session_id") != session_id:
            report.removed_foreign_session += 1
            continue

        try:
            x = float(row["x_px"])
            y = float(row["y_px"])
        except (TypeError, ValueError, KeyError):
            report.warnings.append(f"Skipping summary row with bad coords: {row.get('click_id')}")
            continue

        wp, dist = _nearest_waypoint(x, y, waypoints) if waypoints else (None, float("inf"))
        if not row.get("waypoint_id") and wp and dist <= snap_tolerance_px:
            row["waypoint_id"] = wp.get("waypoint_id", "")
            report.backfilled_waypoint_id += 1

        if drop_off_path and waypoints and dist > snap_tolerance_px:
            report.removed_off_path += 1
            report.warnings.append(
                f"Dropped off-path click {row.get('click_id')} ({dist:.0f}px from nearest waypoint)"
            )
            continue

        click_id = row.get("click_id", "")
        if click_id:
            kept_click_ids.add(click_id)
        kept_summary.append(row)

    # One row per waypoint_id (keep last click if duplicates)
    if waypoints:
        by_wp: dict[str, dict] = {}
        no_wp: List[dict] = []
        for row in kept_summary:
            wid = row.get("waypoint_id") or ""
            if wid:
                by_wp[wid] = row
            else:
                no_wp.append(row)
        kept_summary = no_wp + list(by_wp.values())
        kept_summary.sort(key=lambda r: r.get("click_id", ""))
        kept_click_ids = {r.get("click_id", "") for r in kept_summary if r.get("click_id")}

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in kept_summary:
            writer.writerow(row)
    report.summary_after = len(kept_summary)

    kept_raw = [
        r for r in raw_rows
        if r.get("session_id") == session_id and r.get("click_id") in kept_click_ids
    ]
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in kept_raw:
            writer.writerow(row)
    report.raw_after = len(kept_raw)

    return report


def count_foreign_rows(session_dir: Path) -> int:
    session_id = Path(session_dir).name
    summary_path = Path(session_dir) / "measurements_summary.csv"
    if not summary_path.exists():
        return 0
    with open(summary_path, newline="", encoding="utf-8") as f:
        return sum(1 for row in csv.DictReader(f) if row.get("session_id") != session_id)
