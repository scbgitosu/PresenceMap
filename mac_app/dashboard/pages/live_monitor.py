"""Live → Monitor / Diagnostics."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from mac_app.inference.live_buffer import LiveBuffer
from mac_app.inference.runtime import InferenceConfig, InferenceLoop
from mac_app.train.registry import list_models


def _get_loop() -> InferenceLoop | None:
    return st.session_state.get("inference_loop")


def render(project_dir: Path, *, page: str = "Monitor") -> None:
    if page == "Monitor":
        _render_monitor(project_dir)
    elif page == "Diagnostics":
        _render_diagnostics(project_dir)


def _render_monitor(project_dir: Path) -> None:
    st.header("Live → Monitor")
    loop = _get_loop()
    running = loop is not None

    models = list_models(project_dir)
    if not models and not running:
        st.warning("no trained models yet -- use Training → Train first")
        return
    model_ids = [m.model_id for m in models]

    cols = st.columns([3, 1, 1, 1])
    selected = cols[0].selectbox(
        "model",
        model_ids,
        index=0 if model_ids else None,
        disabled=running,
    )
    windows_endpoint = cols[1].text_input("windows endpoint", value="tcp://localhost:5555")
    auto_refresh = cols[2].checkbox("Auto refresh", value=True)
    if not running and cols[3].button("Start live", type="primary", disabled=not selected):
        cfg = InferenceConfig(
            project_dir=project_dir,
            model_id=selected,
            windows_endpoint=windows_endpoint,
        )
        try:
            loop = InferenceLoop(cfg)
            loop.start()
            st.session_state["inference_loop"] = loop
            st.rerun()
        except Exception as exc:
            st.error(f"could not start: {exc}")
            return
    if running and cols[3].button("Stop live", type="primary"):
        loop.stop()
        st.session_state.pop("inference_loop", None)
        st.success("stopped")
        st.rerun()

    if not running:
        st.info("Press **Start live** to begin scoring incoming windows.")
        return

    stats = loop.stats()
    buf = LiveBuffer()
    latest = buf.latest_prediction()

    cols = st.columns(4)
    state_color = "🟢" if stats.last_state == "vacant" else ("🟡" if stats.last_state == "unknown" else "🔴")
    cols[0].metric("state", f"{state_color} {stats.last_state}")
    cols[1].metric("confidence", f"{stats.last_confidence:.2f}")
    cols[2].metric("motion", f"{stats.last_motion:.2f}")
    cols[3].metric("signal age (s)", f"{stats.last_window_age_s:.1f}")
    if stats.signal_lost:
        st.error("signal lost from agent — is the HP agent running?")

    history = buf.predictions(limit=600)  # last ~20 min @ 2 Hz
    if history:
        df = pd.DataFrame([r.to_dict() for r in history])
        st.subheader("Motion intensity (last ~20 min)")
        st.line_chart(df.set_index("received_at")[["motion_intensity", "confidence"]])
        st.subheader("Recent state changes")
        df["changed"] = df["stable"] != df["stable"].shift(1)
        changes = df[df["changed"]].tail(20)
        if not changes.empty:
            st.dataframe(
                changes[["received_at", "stable", "raw", "confidence", "motion_intensity"]],
                hide_index=True,
                use_container_width=True,
            )
    st.caption(
        f"REST endpoints (default port 8765):  "
        f"GET /state, /history, /health, /models  -- start with `presence-mac api`."
    )
    if auto_refresh:
        time.sleep(1.5)
        st.rerun()


def _render_diagnostics(project_dir: Path) -> None:
    st.header("Live → Diagnostics")
    buf = LiveBuffer()
    rows = buf.recent_health(limit=50)
    if not rows:
        st.info("no agent health messages yet")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

    latest = buf.latest_prediction()
    if latest is not None:
        st.subheader("Most recent window")
        st.json({
            "window_id": latest.window_id,
            "raw": latest.raw,
            "stable": latest.stable,
            "confidence": latest.confidence,
            "motion_intensity": latest.motion_intensity,
            "model_id": latest.model_id,
            "received_at": latest.received_at,
        })
