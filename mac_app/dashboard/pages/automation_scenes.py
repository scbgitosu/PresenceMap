"""Live → Automation scenes (Phase 3 — MQTT / Home Assistant)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def render(project_dir: Path) -> None:
    st.header("Live → Automation scenes")
    st.info(
        "**Phase 3** — Scene training (movie mode, bedtime, etc.) and MQTT / Home Assistant "
        "outputs are planned after bedroom presence is stable. "
        "See [docs/ROADMAP.md](../../../docs/ROADMAP.md)."
    )
    st.markdown(
        "Planned flow:\n"
        "- Record CSI while you label a scene (e.g. `scene_movie`, `scene_bedtime`).\n"
        "- Train a small classifier on fused ESP32 window features.\n"
        "- Emit webhooks or MQTT topics when confidence exceeds a threshold.\n"
    )
    st.checkbox("Enable MQTT output (not implemented)", disabled=True)
    st.text_input("Home Assistant webhook URL", disabled=True, placeholder="http://homeassistant:8123/api/webhook/...")
