"""Training → Review page: per-window table + thumbnail grid."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from mac_app.train.dataset import list_v2_sessions, read_session_parquet
from shared.project import session_paths


def render(project_dir: Path) -> None:
    st.header("Training → Review")
    sessions = list_v2_sessions(project_dir)
    if not sessions:
        st.info("no v2 sessions yet")
        return
    session_id = st.selectbox("session", sessions)
    paths = session_paths(project_dir, session_id)

    rf = read_session_parquet(project_dir, session_id, kind="rf_windows")
    labels = read_session_parquet(project_dir, session_id, kind="labels")
    yolo = read_session_parquet(project_dir, session_id, kind="yolo_frames")
    thumbs = read_session_parquet(project_dir, session_id, kind="thumbnails")

    cols = st.columns(4)
    cols[0].metric("RF windows", 0 if rf is None else len(rf))
    cols[1].metric("YOLO frames", 0 if yolo is None else len(yolo))
    cols[2].metric("labels", 0 if labels is None else len(labels))
    cols[3].metric("thumbnails", 0 if thumbs is None else len(thumbs))

    if rf is not None and labels is not None and not rf.empty:
        merged = rf.merge(
            labels[["window_id", "occupancy", "person_count", "label_confidence", "label_source"]],
            on="window_id",
            how="left",
        )
        st.subheader("Windows + labels")
        st.dataframe(
            merged[
                [
                    "window_id",
                    "phase",
                    "target_rssi_avg_dbm",
                    "snr_db",
                    "target_seen",
                    "csi_frames",
                    "csi_loss_ratio",
                    "occupancy",
                    "person_count",
                    "label_confidence",
                    "label_source",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    if thumbs is not None and not thumbs.empty:
        st.subheader("Thumbnails")
        kept = thumbs[thumbs["kept_reason"] != "purged"].head(12)
        if not kept.empty:
            grid = st.columns(4)
            for i, row in enumerate(kept.itertuples(index=False)):
                p = paths["session_dir"] / row.path
                if p.exists():
                    grid[i % 4].image(str(p), caption=f"persons={row.person_count}")
