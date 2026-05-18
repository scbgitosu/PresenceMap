"""Project directory path resolution.

Single source of truth for where files live within a project directory.
Used by both ``hp_agent`` and ``mac_app``. v2-aware: knows about the new
``sessions/`` layout under each project alongside the v1 ``presence_sessions/``
and ``survey_sessions/`` directories that remain for the legacy viewer.
"""
from __future__ import annotations

from pathlib import Path


def project_paths(project_dir: Path | str) -> dict:
    """Resolve standard subpaths from a project root directory.

    The project root is typically ``data/survey_projects/<name>/``.
    """
    d = Path(project_dir)
    return {
        "project_dir": d,
        "floorplan_png": d / "floorplan.png",
        "floorplan_metadata": d / "floorplan_metadata.json",
        "project_config": d / "project_config.json",
        "rooms_json": d / "rooms.json",
        "router_positions_json": d / "router_positions.json",
        "walk_waypoints_json": d / "walk_waypoints.json",
        # v2 sessions
        "sessions_dir": d / "sessions",
        # v1 (read-only via shared.legacy_v1)
        "presence_sessions_dir": d / "presence_sessions",
        "survey_sessions_dir": d / "survey_sessions",
    }


def session_paths(project_dir: Path | str, session_id: str) -> dict:
    """Resolve standard file paths for a v2 session."""
    p = project_paths(project_dir)
    s = p["sessions_dir"] / session_id
    return {
        "session_dir": s,
        "session_json": s / "session.json",
        "rf_windows_parquet": s / "rf_windows.parquet",
        "labels_parquet": s / "labels.parquet",
        "yolo_frames_parquet": s / "yolo_frames.parquet",
        "thumbnails_dir": s / "thumbnails",
        "thumbnails_index_parquet": s / "thumbnails" / "index.parquet",
        "health_parquet": s / "health.parquet",
        "agent_log": s / "agent.log.jsonl",
        "mac_log": s / "mac.log.jsonl",
    }


def models_dir(repo_root: Path | str) -> Path:
    """Where trained models live: ``<repo_root>/data/models/``."""
    return Path(repo_root) / "data" / "models"


def runtime_dir(repo_root: Path | str) -> Path:
    """Where live-mode runtime state (e.g. live_buffer.sqlite3) lives."""
    return Path(repo_root) / "data" / "runtime"
