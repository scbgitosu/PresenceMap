"""Live → Room occupancy (Phase 2 — requires distributed ESP32 placement)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from shared.project import project_paths


def render(project_dir: Path) -> None:
    st.header("Live → Room occupancy")
    st.warning(
        "**Phase 2** — Your current plan uses three nodes in one bedroom. "
        "Per-room occupancy needs nodes spread across rooms (living, kitchen, hallway) "
        "aligned with `rooms.json` on the floorplan."
    )
    p = project_paths(project_dir)
    if p["rooms_json"].exists():
        st.subheader("Configured rooms")
        st.json(__import__("json").loads(p["rooms_json"].read_text()), expanded=False)
    st.markdown(
        "Next steps when hardware is redeployed:\n"
        "1. Place one ESP32 per zone with line-of-sight to that room.\n"
        "2. Record labeled sessions per room.\n"
        "3. Train multi-head or per-room models.\n"
        "4. Enable floorplan heatmap in this page."
    )
