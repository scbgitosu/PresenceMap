"""Read v1 PresenceMap session CSVs into pandas DataFrames.

These are the on-disk artifacts produced by the deleted ``hp_collector``
pipeline. The functions here are read-only and best-effort: missing files
return empty DataFrames rather than raising, so the Legacy Viewer never
crashes on a partial session.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

V1_FEATURE_FILE = "presence_features.csv"
V1_LABEL_FILE = "presence_labels.csv"
V1_EVENT_FILE = "presence_events.csv"
V1_STATE_FILE = "presence_states.csv"
V1_HEALTH_FILE = "presence_health.csv"
V1_RAW_FILE = "presence_raw.csv"
V1_WEBCAM_EVENTS_FILE = "presence_webcam_events.csv"
V1_WEBCAM_HEALTH_FILE = "presence_webcam_health.csv"
V1_EXPERIMENT_FILE = "presence_experiment.json"


def list_v1_sessions(project_dir: Path | str) -> List[Path]:
    """List v1 session directories under ``<project>/presence_sessions/``."""
    d = Path(project_dir) / "presence_sessions"
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir())


def _read_csv(session_dir: Path, name: str) -> pd.DataFrame:
    p = Path(session_dir) / name
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def load_v1_features(session_dir: Path | str) -> pd.DataFrame:
    return _read_csv(Path(session_dir), V1_FEATURE_FILE)


def load_v1_labels(session_dir: Path | str) -> pd.DataFrame:
    return _read_csv(Path(session_dir), V1_LABEL_FILE)


def load_v1_events(session_dir: Path | str) -> pd.DataFrame:
    return _read_csv(Path(session_dir), V1_EVENT_FILE)


def load_v1_states(session_dir: Path | str) -> pd.DataFrame:
    return _read_csv(Path(session_dir), V1_STATE_FILE)


def load_v1_health(session_dir: Path | str) -> pd.DataFrame:
    return _read_csv(Path(session_dir), V1_HEALTH_FILE)


def load_v1_webcam_events(session_dir: Path | str) -> pd.DataFrame:
    return _read_csv(Path(session_dir), V1_WEBCAM_EVENTS_FILE)


def load_v1_webcam_health(session_dir: Path | str) -> pd.DataFrame:
    return _read_csv(Path(session_dir), V1_WEBCAM_HEALTH_FILE)


def load_v1_experiment(session_dir: Path | str) -> dict:
    """Read presence_experiment.json. Returns {} if missing or malformed."""
    import json

    p = Path(session_dir) / V1_EXPERIMENT_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}
