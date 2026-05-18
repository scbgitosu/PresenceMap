"""Overview page: project status + session counts."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from mac_app.train.dataset import list_v2_sessions
from shared.legacy_v1.readers import list_v1_sessions
from shared.project import project_paths


def render(project_dir: Path) -> None:
    st.header("Overview")
    p = project_paths(project_dir)
    cfg_path = p["project_config"]
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}
    cols = st.columns(3)
    cols[0].metric("Project", project_dir.name)
    cols[1].metric("Target SSID", cfg.get("target_ssid", "—"))
    cols[2].metric("Default interface", cfg.get("default_interface", "—"))

    st.subheader("Sessions")
    v2 = list_v2_sessions(project_dir)
    v1 = [s.name for s in list_v1_sessions(project_dir)]
    cols = st.columns(2)
    with cols[0]:
        st.caption(f"v2 ({len(v2)})")
        if v2:
            st.write(v2)
        else:
            st.info("no v2 sessions yet -- create one in Training → Collect")
    with cols[1]:
        st.caption(f"v1 (legacy, {len(v1)})")
        if v1:
            st.write(v1)
        else:
            st.info("no v1 sessions on disk")

    st.subheader("Floorplan")
    fp = p["floorplan_png"]
    if fp.exists():
        st.image(str(fp), caption=f"{fp.name}", use_container_width=True)
    else:
        st.info("no floorplan -- run `streamlit run mac_app/floorplan/floorplan_import.py`")
