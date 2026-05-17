"""PresenceMap Streamlit dashboard for setup, training, and review.

Usage:
    streamlit run mac_analysis/survey_dashboard.py -- --project survey_projects/apartment_test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_mac_analysis = Path(__file__).resolve().parent
_repo_root = _mac_analysis.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_mac_analysis))

import streamlit_canvas_compat  # noqa: F401 - patch Streamlit before canvas modules

import pandas as pd
import streamlit as st

from mac_analysis.presence_training import train_and_evaluate
from shared.utils import project_paths
from streamlit_project_cli import parse_streamlit_project_args


def _parse_args():
    return parse_streamlit_project_args()


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _status(label: str, ok: bool, detail: str = ""):
    icon = "OK" if ok else "Missing"
    st.write(f"**{label}:** {icon}{' - ' + detail if detail else ''}")


def _json_if_exists(path: Path):
    if path and path.exists():
        st.json(_load_json(path, {}))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _presence_sessions(project_dir: Path) -> list[str]:
    root = project_dir / "presence_sessions"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _hp_target(project_dir: Path) -> str:
    cfg = _load_json(project_paths(project_dir)["project_config"], {})
    return cfg.get("presence", {}).get("hp_sync_target", "user@hp-laptop:~/PresenceMap").rstrip("/")


def _render_overview(project_dir: Path, paths: dict):
    st.subheader("PresenceMap Readiness")
    cfg = _load_json(paths["project_config"], {})
    presence = cfg.get("presence", {})
    sessions = _presence_sessions(project_dir)

    _status("Presence config", paths["project_config"].exists(), str(paths["project_config"]))
    _status("Target SSID", bool(cfg.get("target_ssid")), cfg.get("target_ssid", "needed for RF collection"))
    _status("Default interface", bool(cfg.get("default_interface")), cfg.get("default_interface", "can also be set on HP"))
    _status("Default session", bool(presence.get("default_session")), presence.get("default_session", "set in Setup"))
    _status("Presence sessions", bool(sessions), f"{len(sessions)} found")

    st.subheader("Optional Context")
    _status("Floorplan", paths["floorplan_png"].exists(), str(paths["floorplan_png"]))
    _status("Rooms", paths["rooms_json"].exists(), str(paths["rooms_json"]))
    st.caption("Floorplan files are optional context for PresenceMap, not a blocker for RF collection.")


def _render_setup(project_dir: Path):
    st.subheader("Presence Room Experiment")
    paths = project_paths(project_dir)
    existing = _load_json(paths["project_config"], {})
    presence_cfg = existing.get("presence", {})

    with st.form("presence_quick_setup"):
        project_name = st.text_input("Project name", value=existing.get("project_name", project_dir.name))
        target_ssid = st.text_input("Target SSID", value=existing.get("target_ssid", ""))
        target_bssid = st.text_input("Target BSSID (optional)", value=existing.get("target_bssid", ""))
        default_interface = st.text_input("HP Wi-Fi interface", value=existing.get("default_interface", "wlan1"))
        scan_backend = st.selectbox(
            "Scan backend",
            ["iw", "auto", "nmcli"],
            index=["iw", "auto", "nmcli"].index(existing.get("scan_backend", "iw"))
            if existing.get("scan_backend", "iw") in ["iw", "auto", "nmcli"]
            else 0,
        )
        room_name = st.text_input("First room label", value=presence_cfg.get("first_room_label", "bedroom"))
        default_session = st.text_input("Default presence session", value=presence_cfg.get("default_session", "bedroom_v1"))
        hp_target = st.text_input("HP sync target", value=presence_cfg.get("hp_sync_target", "user@hp-laptop:~/PresenceMap"))
        submitted = st.form_submit_button("Initialize / Save Presence Project", type="primary")

    if submitted:
        config = {
            "project_name": project_name,
            "target_ssid": target_ssid,
            "target_bssid": target_bssid,
            "default_interface": default_interface,
            "units": existing.get("units", "feet"),
            "collection_mode": "presence_room_experiment",
            "scan_backend": scan_backend,
            "presence": {
                "collector": "HP Linux laptop + AR9271",
                "analysis": "MacBook training/review",
                "webcam_role": "derived_ground_truth_labels",
                "video_retention": "derived_labels_only",
                "first_room_label": room_name,
                "default_session": default_session,
                "hp_sync_target": hp_target,
                "label_source": "webcam_derived",
                "label_source_detail": "derived occupancy only; no continuous video retained",
                "collector_placement": f"HP + AR9271 fixed in {room_name}",
                "router_placement": f"router fixed in {room_name}",
            },
            "paths": {
                "floorplan_png": str(paths["floorplan_png"]),
                "rooms_json": str(paths["rooms_json"]),
                "router_positions_json": str(paths["router_positions_json"]),
            },
        }
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "presence_sessions" / default_session).mkdir(parents=True, exist_ok=True)
        _ensure_file(project_dir / "presence_sessions" / ".gitkeep")
        _write_json(paths["project_config"], config)
        st.success(f"Initialized Presence project at `{project_dir}`")

    hp_target = _hp_target(project_dir)
    default_session = presence_cfg.get("default_session", "bedroom_v1")
    hp_project = f"{hp_target}/survey_projects/{project_dir.name}"
    st.subheader("Sync Commands")
    st.code(f"rsync -av {project_dir}/ {hp_project}/")
    st.code(f"rsync -av {hp_project}/presence_sessions/ {project_dir}/presence_sessions/")
    st.code(f"python3 mac_analysis/presence_training.py --project {project_dir} --session {default_session}")

    st.subheader("Optional Floorplan")
    st.caption("Use these later only if you want map overlays.")
    st.code(f"streamlit run mac_analysis/floorplan_import.py -- --project {project_dir}")
    st.code(f"streamlit run mac_analysis/floorplan_labeler.py -- --project {project_dir}")


def _render_presence_occupancy(project_dir: Path):
    st.subheader("Presence / Occupancy")
    session_ids = _presence_sessions(project_dir)
    if not session_ids:
        st.info("No `presence_sessions/*` folders found yet. Use Setup, sync to HP, then run the HP launcher.")
        return

    session_id = st.selectbox("Presence session", session_ids, key="presence_session")
    session_dir = project_dir / "presence_sessions" / session_id
    features_path = session_dir / "presence_features.csv"
    labels_path = session_dir / "presence_labels.csv"
    states_path = session_dir / "presence_states.csv"
    events_path = session_dir / "presence_events.csv"
    model_path = session_dir / "presence_model.json"
    eval_path = session_dir / "presence_eval.json"
    errors_path = session_dir / "presence_eval_errors.csv"
    health_path = session_dir / "presence_health.csv"
    metadata_path = session_dir / "presence_experiment.json"

    st.caption(f"Artifacts: `{session_dir}`")
    cols = st.columns(7)
    cols[0].metric("Features", "yes" if features_path.exists() else "missing")
    cols[1].metric("Labels", "yes" if labels_path.exists() else "missing")
    cols[2].metric("States", "yes" if states_path.exists() else "missing")
    cols[3].metric("Events", "yes" if events_path.exists() else "missing")
    cols[4].metric("Model", "yes" if model_path.exists() else "missing")
    cols[5].metric("Eval", "yes" if eval_path.exists() else "missing")
    cols[6].metric("Health", "yes" if health_path.exists() else "missing")

    labels_df = _read_csv(labels_path)
    features_df = _read_csv(features_path)
    states_df = _read_csv(states_path)
    events_df = _read_csv(events_path)
    errors_df = _read_csv(errors_path)
    health_df = _read_csv(health_path)

    train_disabled = features_df.empty or labels_df.empty
    if st.button("Train / Evaluate Occupancy Model", type="primary", disabled=train_disabled, key="presence_train_eval_btn"):
        try:
            evaluation = train_and_evaluate(project=project_dir, session=session_id)
        except Exception as e:
            st.error(str(e))
        else:
            st.success("Training evaluation complete.")
            st.json(evaluation)
            st.rerun()
    if train_disabled:
        st.caption("Collect and sync feature plus label rows before training/evaluation.")

    if metadata_path.exists():
        with st.expander("Experiment Metadata", expanded=False):
            _json_if_exists(metadata_path)

    if not labels_df.empty:
        st.subheader("Label Coverage")
        label_counts = labels_df.groupby(["label", "occupancy_label"], dropna=False).size().reset_index(name="windows")
        st.dataframe(label_counts, use_container_width=True, hide_index=True)

    if model_path.exists():
        with st.expander("Occupancy Model", expanded=True):
            _json_if_exists(model_path)

    if eval_path.exists():
        evaluation = _load_json(eval_path, {})
        st.subheader("Training Evaluation")
        cols = st.columns(5)
        cols[0].metric("Model Ready", "yes" if evaluation.get("model_ready") else "no")
        cols[1].metric("Accuracy", evaluation.get("accuracy", "n/a"))
        cols[2].metric("Known Rate", evaluation.get("known_rate", "n/a"))
        cols[3].metric("Errors", evaluation.get("error_count", "n/a"))
        cols[4].metric("Windows", evaluation.get("total_labeled_windows", "n/a"))
        if not errors_df.empty:
            st.subheader("False Positives / False Negatives")
            st.dataframe(errors_df, use_container_width=True, hide_index=True)

    if not states_df.empty:
        st.subheader("State Timeline")
        st.dataframe(states_df.tail(100), use_container_width=True, hide_index=True)
        if {"timestamp_start", "confidence"}.issubset(states_df.columns):
            chart_df = states_df[["timestamp_start", "confidence"]].copy()
            chart_df["timestamp_start"] = chart_df["timestamp_start"].astype(str)
            st.line_chart(chart_df, x="timestamp_start", y="confidence")

    if not features_df.empty:
        st.subheader("Feature Review")
        numeric_candidates = [
            "rssi_avg_dbm",
            "rssi_std_db",
            "snr_avg_db",
            "visible_bssid_count",
            "motion_score",
            "rssi_recent_std_db",
            "recent_motion_score",
        ]
        available = [col for col in numeric_candidates if col in features_df.columns]
        selected = st.multiselect("Feature traces", available, default=available[:3], key="presence_features")
        if selected and "timestamp_start" in features_df.columns:
            chart_df = features_df[["timestamp_start"] + selected].copy()
            chart_df["timestamp_start"] = chart_df["timestamp_start"].astype(str)
            st.line_chart(chart_df, x="timestamp_start", y=selected)
        st.dataframe(features_df.tail(100), use_container_width=True, hide_index=True)

    if not events_df.empty:
        st.subheader("Motion Events")
        st.dataframe(events_df.tail(100), use_container_width=True, hide_index=True)

    if not health_df.empty:
        st.subheader("Collector Health")
        health_counts = health_df.groupby(["phase", "status"], dropna=False).size().reset_index(name="windows")
        st.dataframe(health_counts, use_container_width=True, hide_index=True)
        st.dataframe(health_df.tail(100), use_container_width=True, hide_index=True)


def _render_hp_commands(project_dir: Path):
    st.subheader("Transfer and Collect on HP")
    hp_target = _hp_target(project_dir)
    hp_project = f"{hp_target}/survey_projects/{project_dir.name}"
    st.code(f"rsync -av {project_dir}/ {hp_project}/")
    st.code(f"python3 hp_collector/preflight.py --project survey_projects/{project_dir.name}")
    st.code(f"python3 hp_collector/collector_launcher.py --project survey_projects/{project_dir.name}")
    st.code(f"rsync -av {hp_project}/presence_sessions/ {project_dir}/presence_sessions/")


def main():
    args = _parse_args()
    project_dir = Path(args.project)
    paths = project_paths(project_dir)

    st.set_page_config(page_title="PresenceMap Dashboard", layout="wide")
    st.title("PresenceMap Dashboard")
    st.caption(f"Project: `{project_dir}`")

    tabs = st.tabs([
        "Overview",
        "Setup",
        "Presence/Occupancy",
        "HP Transfer/Collect",
    ])
    with tabs[0]:
        _render_overview(project_dir, paths)
    with tabs[1]:
        _render_setup(project_dir)
    with tabs[2]:
        _render_presence_occupancy(project_dir)
    with tabs[3]:
        _render_hp_commands(project_dir)


if __name__ == "__main__":
    main()
