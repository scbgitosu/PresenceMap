"""Legacy Viewer page: read-only view of v1 presence_sessions/ CSVs."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from shared.legacy_v1.readers import (
    list_v1_sessions,
    load_v1_experiment,
    load_v1_features,
    load_v1_health,
    load_v1_labels,
    load_v1_webcam_events,
    load_v1_webcam_health,
)


def render(project_dir: Path) -> None:
    st.header("Legacy Viewer (v1, read-only)")
    sessions = list_v1_sessions(project_dir)
    if not sessions:
        st.info("no v1 sessions under presence_sessions/")
        return
    pick = st.selectbox("session", [s.name for s in sessions])
    session_dir = next(s for s in sessions if s.name == pick)
    st.caption(str(session_dir))

    exp = load_v1_experiment(session_dir)
    if exp:
        with st.expander("presence_experiment.json"):
            st.json(exp)

    features = load_v1_features(session_dir)
    labels = load_v1_labels(session_dir)
    cols = st.columns(2)
    cols[0].metric("feature windows", len(features))
    cols[1].metric("labels", len(labels))

    if not features.empty:
        st.subheader("Feature traces")
        candidates = [c for c in ["rssi_avg_dbm", "snr_avg_db", "motion_score"] if c in features.columns]
        if candidates:
            st.line_chart(features[candidates])

    if not labels.empty:
        st.subheader("Label distribution")
        if "occupancy_label" in labels.columns:
            st.bar_chart(labels["occupancy_label"].value_counts())

    health = load_v1_health(session_dir)
    if not health.empty:
        with st.expander("RF health"):
            st.dataframe(health.tail(40), hide_index=True, use_container_width=True)

    webcam_events = load_v1_webcam_events(session_dir)
    webcam_health = load_v1_webcam_health(session_dir)
    if not webcam_events.empty or not webcam_health.empty:
        with st.expander("Webcam (motion-based v1 ground truth)"):
            if not webcam_events.empty:
                st.dataframe(webcam_events.tail(40), hide_index=True, use_container_width=True)
            if not webcam_health.empty:
                st.dataframe(webcam_health.tail(40), hide_index=True, use_container_width=True)
