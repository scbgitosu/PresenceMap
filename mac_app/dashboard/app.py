"""PresenceMap dashboard — ESP32 CSI sensing."""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import streamlit as st

from mac_app.dashboard._cli import parse_streamlit_project_args
from mac_app.dashboard.pages import (
    automation_scenes,
    esp32_diagnostics,
    models_browser,
    overview,
    room_occupancy,
    training_train,
)


def _resolve_project(args) -> Path:
    p = Path(args.project)
    if not p.is_absolute():
        p = (_repo_root / p).resolve()
    return p


def main() -> None:
    st.set_page_config(page_title="PresenceMap", layout="wide")
    args = parse_streamlit_project_args()
    project_dir = _resolve_project(args)

    with st.sidebar:
        st.title("PresenceMap")
        st.caption(f"project: `{project_dir.name}`")
        page = st.radio(
            "Page",
            options=[
                "Overview",
                "ESP32 Nodes",
                "Train",
                "Models",
                "Room occupancy",
                "Automation",
            ],
            index=1,
        )

    if page == "Overview":
        overview.render(project_dir)
    elif page == "ESP32 Nodes":
        esp32_diagnostics.render(project_dir)
    elif page == "Train":
        training_train.render(project_dir)
    elif page == "Models":
        models_browser.render(project_dir)
    elif page == "Room occupancy":
        room_occupancy.render(project_dir)
    elif page == "Automation":
        automation_scenes.render(project_dir)


if __name__ == "__main__":
    main()
