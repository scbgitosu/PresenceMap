"""Live mode placeholder (Stage 6)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def render(project_dir: Path, *, page: str = "Monitor") -> None:
    st.header(f"Live → {page}")
    st.info(
        "Stage 6 lands the live inference runtime, the state card, and the FastAPI "
        "REST endpoints (`/state`, `/history`, `/health`)."
    )
