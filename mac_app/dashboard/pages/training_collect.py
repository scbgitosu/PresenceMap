"""Training → Collect page: start/stop a session and watch it grow."""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from mac_app.recorder import RecorderConfig, SessionRecorder

PHASES = ["calibration", "labeled_vacant", "labeled_occupied", "freeform"]


def _get_recorder() -> SessionRecorder | None:
    return st.session_state.get("recorder")


def render(project_dir: Path) -> None:
    st.header("Training → Collect")

    rec = _get_recorder()
    running = rec is not None

    cols = st.columns([3, 1, 1])
    session_id = cols[0].text_input(
        "session_id",
        value=st.session_state.get("session_id", "train001"),
        disabled=running,
    )
    enable_yolo = cols[1].checkbox(
        "YOLO labeling", value=st.session_state.get("enable_yolo", True), disabled=running
    )
    auto_refresh = cols[2].checkbox(
        "Auto refresh", value=st.session_state.get("auto_refresh", True),
    )

    cols = st.columns([1, 1, 1, 1])
    windows_endpoint = cols[0].text_input("windows endpoint", value="tcp://localhost:5555")
    frames_endpoint = cols[1].text_input("frames endpoint", value="tcp://localhost:5556")
    phase = cols[2].selectbox("phase", PHASES, index=PHASES.index(st.session_state.get("phase", "freeform")))
    if running and phase != st.session_state.get("phase"):
        rec.set_phase(phase)
        st.session_state["phase"] = phase

    if not running:
        if cols[3].button("Start session", type="primary"):
            st.session_state["session_id"] = session_id
            st.session_state["enable_yolo"] = enable_yolo
            st.session_state["phase"] = phase
            st.session_state["auto_refresh"] = auto_refresh
            cfg = RecorderConfig(
                project_dir=project_dir,
                session_id=session_id,
                windows_endpoint=windows_endpoint,
                frames_endpoint=frames_endpoint,
                enable_yolo=enable_yolo,
                initial_phase=phase,
            )
            rec = SessionRecorder(cfg)
            rec.start()
            st.session_state["recorder"] = rec
            st.rerun()
    else:
        if cols[3].button("Stop session", type="primary"):
            rec.stop()
            st.success(f"session `{rec.cfg.session_id}` finalized")
            st.session_state.pop("recorder", None)
            st.rerun()

    if running:
        stats = rec.stats()
        st.subheader("Live")
        m = st.columns(5)
        m[0].metric("windows", stats.windows)
        m[1].metric("frames", stats.frames)
        m[2].metric("max persons", stats.persons_seen_max)
        m[3].metric("last YOLO conf", f"{stats.last_yolo_conf:.2f}")
        m[4].metric("signal age (s)", f"{stats.last_window_age_s:.1f}")
        if stats.signal_lost:
            st.error("signal lost from agent — is `presence-agent run` running?")
        snap = rec.snapshot()
        if not snap["rf_windows"].empty:
            recent = snap["rf_windows"].tail(8)[
                ["window_id", "phase", "target_rssi_avg_dbm", "snr_db", "target_seen", "csi_frames", "csi_loss_ratio"]
            ]
            st.dataframe(recent, hide_index=True, use_container_width=True)
        if not snap["labels"].empty:
            label_counts = snap["labels"]["occupancy"].value_counts().to_dict()
            st.caption(f"labels so far: {label_counts}")
        if auto_refresh:
            time.sleep(1.5)
            st.rerun()
    else:
        st.info("Press **Start session** to begin recording.")
