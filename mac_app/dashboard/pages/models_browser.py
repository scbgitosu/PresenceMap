"""Models page placeholder (Stage 5 fills in train + compare)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def render(project_dir: Path) -> None:
    st.header("Models")
    st.info(
        "Stage 5 lands the trainer + model registry here. Until then, train models "
        "via `presence-mac train --session <id>` (also lands in Stage 5)."
    )
