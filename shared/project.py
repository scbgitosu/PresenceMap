"""Project directory path resolution."""
from __future__ import annotations

from pathlib import Path


def project_paths(project_dir: Path | str) -> dict:
    d = Path(project_dir)
    return {
        "project_dir": d,
        "project_config": d / "project_config.json",
        "rooms_json": d / "rooms.json",
        "sessions_dir": d / "sessions",
    }


def session_paths(project_dir: Path | str, session_id: str) -> dict:
    p = project_paths(project_dir)
    s = p["sessions_dir"] / session_id
    return {
        "session_dir": s,
        "session_json": s / "session.json",
        "rf_windows_parquet": s / "rf_windows.parquet",
        "labels_parquet": s / "labels.parquet",
    }


def models_dir(repo_root: Path | str) -> Path:
    return Path(repo_root) / "data" / "models"


def runtime_dir(repo_root: Path | str) -> Path:
    return Path(repo_root) / "data" / "runtime"
