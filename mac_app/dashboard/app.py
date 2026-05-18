"""PresenceMap v2 Streamlit dashboard.

Single entry point with a Training / Live toggle in the sidebar. The Live
pages are stubs until Stage 6 wires the live inference runtime and FastAPI
endpoints.

Launch:
    streamlit run mac_app/dashboard/app.py -- --project data/survey_projects/apartment_test

Or via the CLI:
    presence-mac dashboard --project data/survey_projects/apartment_test
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import streamlit as st

from mac_app.dashboard.pages import (
    legacy_viewer,
    live_monitor,
    models_browser,
    overview,
    training_collect,
    training_review,
    training_setup,
    training_train,
)
from mac_app.floorplan._cli import parse_streamlit_project_args


def _resolve_project(args) -> Path:
    p = Path(args.project)
    if not p.is_absolute():
        p = (_repo_root / p).resolve()
    return p


def main() -> None:
    st.set_page_config(page_title="PresenceMap v2", layout="wide")
    args = parse_streamlit_project_args(
        default_project="data/survey_projects/apartment_test"
    )
    project_dir = _resolve_project(args)

    with st.sidebar:
        st.title("PresenceMap")
        st.caption(f"project: `{project_dir.name}`")
        mode = st.radio("Mode", options=["Training", "Live"], index=0, horizontal=True)
        if mode == "Training":
            page = st.radio(
                "Page",
                options=[
                    "Overview",
                    "Setup",
                    "Collect",
                    "Review",
                    "Train",
                    "Models",
                    "Legacy Viewer",
                ],
                index=0,
            )
        else:
            page = st.radio(
                "Page",
                options=["Monitor", "Diagnostics"],
                index=0,
            )

    if mode == "Training":
        if page == "Overview":
            overview.render(project_dir)
        elif page == "Setup":
            training_setup.render(project_dir)
        elif page == "Collect":
            training_collect.render(project_dir)
        elif page == "Review":
            training_review.render(project_dir)
        elif page == "Train":
            training_train.render(project_dir)
        elif page == "Models":
            models_browser.render(project_dir)
        elif page == "Legacy Viewer":
            legacy_viewer.render(project_dir)
    else:
        live_monitor.render(project_dir, page=page)


if __name__ == "__main__":
    main()
