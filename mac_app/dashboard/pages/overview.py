"""Project overview."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from mac_app.train.dataset import list_sessions
from shared.project import project_paths


def render(project_dir: Path) -> None:
    st.header("Overview")
    st.markdown(
        "PresenceMap uses **ESP32-S3** nodes (RuView firmware) streaming CSI to your Mac. "
        "Start with **ESP32 Nodes** diagnostics, then record labeled sessions and train a "
        "small occupancy model."
    )
    p = project_paths(project_dir)
    if p["project_config"].exists():
        st.subheader("Project config")
        st.json(json.loads(p["project_config"].read_text()), expanded=False)
    sessions = list_sessions(project_dir)
    st.subheader("Recorded sessions")
    if sessions:
        st.write(", ".join(f"`{s}`" for s in sessions))
    else:
        st.caption("No sessions yet — record from **ESP32 Nodes** or `presence-mac esp32-record`.")
    st.code(f"presence-mac esp32-ingest --project {project_dir}", language="bash")
