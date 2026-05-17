"""
Unified Streamlit control panel for the Mac-side Wi-Fi survey workflow.

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

from mac_analysis.heatmap_generator import run_heatmap_generation
from mac_analysis.placement_optimizer import optimize_placement
from mac_analysis.session_compare import run_session_comparison
from shared.survey_metrics import DEFAULT_GOOD_DBM, DEFAULT_WEAK_DBM, discover_sessions
from shared.utils import project_paths
from streamlit_project_cli import parse_streamlit_project_args


def _parse_args():
    return parse_streamlit_project_args()


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _project_output_dir(project_dir: Path) -> Path:
    return project_dir / "output"


def _status(label: str, ok: bool, detail: str = ""):
    icon = "OK" if ok else "Missing"
    st.write(f"**{label}:** {icon}{' - ' + detail if detail else ''}")


def _image_if_exists(path: Path, caption: str | None = None):
    if path and path.exists():
        st.image(str(path), caption=caption or path.name, use_container_width=True)


def _dataframe_if_exists(path: Path):
    if path and path.exists():
        st.dataframe(pd.read_csv(path), use_container_width=True)


def _json_if_exists(path: Path):
    if path and path.exists():
        st.json(_load_json(path, {}))


def _session_selector(session_ids: list[str], *, key: str, multiselect: bool):
    if multiselect:
        default = session_ids[: min(3, len(session_ids))]
        return st.multiselect("Sessions", session_ids, default=default, key=key)
    if not session_ids:
        return None
    return st.selectbox("Session", session_ids, key=key)


def _render_overview(project_dir: Path, paths: dict, session_ids: list[str]):
    st.subheader("Project Health")
    metadata = _load_json(paths["floorplan_metadata"], {})
    _status("Floorplan", paths["floorplan_png"].exists(), str(paths["floorplan_png"]))
    _status("Metadata", paths["floorplan_metadata"].exists(), str(paths["floorplan_metadata"]))
    _status(
        "Scale",
        bool(metadata.get("scale_pixels_per_foot")),
        f"{metadata.get('scale_pixels_per_foot')} px/ft" if metadata.get("scale_pixels_per_foot") else "required for placement optimizer",
    )
    _status("Rooms", paths["rooms_json"].exists(), str(paths["rooms_json"]))
    _status("Router positions", paths["router_positions_json"].exists(), str(paths["router_positions_json"]))
    _status("Walk waypoints", paths["walk_waypoints_json"].exists(), "optional but recommended")
    _status("Survey sessions", bool(session_ids), f"{len(session_ids)} found")

    st.subheader("Detected Sessions")
    if session_ids:
        st.write(session_ids)
    else:
        st.info("No `survey_sessions/*/measurements_summary.csv` files found yet.")

    output_dir = _project_output_dir(project_dir)
    st.subheader("Output Folder")
    st.code(str(output_dir))


def _render_setup(project_dir: Path):
    st.subheader("Setup Apps")
    st.write("These setup steps still use the existing canvas-heavy Streamlit apps.")
    st.code(f"streamlit run mac_analysis/floorplan_import.py -- --project {project_dir}")
    st.code(f"streamlit run mac_analysis/floorplan_labeler.py -- --project {project_dir}")
    st.caption("Use floorplan import to measure one known wall before running placement optimization.")


def _render_heatmaps(project_dir: Path, session_ids: list[str]):
    st.subheader("Generate Heatmaps")
    if not session_ids:
        st.warning("Collect or transfer at least one survey session first.")
        return

    session_id = _session_selector(session_ids, key="heatmap_session", multiselect=False)
    weak_threshold = st.number_input(
        "Weak threshold (dBm)",
        value=float(DEFAULT_WEAK_DBM),
        step=1.0,
        key="heatmap_weak_threshold",
    )
    output_dir = _project_output_dir(project_dir) / "heatmaps"
    st.caption(f"Outputs: `{output_dir}`")

    if st.button("Generate Heatmaps", type="primary", key="heatmap_generate_btn"):
        try:
            result = run_heatmap_generation(project_dir, session_id, output_dir, weak_threshold)
        except Exception as e:
            st.error(str(e))
            return
        st.success("Heatmaps generated.")
        st.session_state["last_heatmap_result"] = result

    result = st.session_state.get("last_heatmap_result")
    if result:
        st.write(result["stats"])
        for name, path in result["outputs"].items():
            _image_if_exists(path, name)


def _render_comparison(project_dir: Path, session_ids: list[str]):
    st.subheader("Compare Sessions")
    if len(session_ids) < 2:
        st.warning("Need at least two sessions to compare router placements.")
        return

    selected = _session_selector(session_ids, key="compare_sessions", multiselect=True)
    good_threshold = st.number_input(
        "Good threshold (dBm)",
        value=float(DEFAULT_GOOD_DBM),
        step=1.0,
        key="compare_good_threshold",
    )
    weak_threshold = st.number_input(
        "Weak threshold (dBm)",
        value=float(DEFAULT_WEAK_DBM),
        step=1.0,
        key="compare_weak_threshold",
    )
    export_heatmaps = st.checkbox(
        "Also export per-session heatmaps",
        value=True,
        key="compare_export_heatmaps",
    )
    output_dir = _project_output_dir(project_dir) / "comparison"
    st.caption(f"Outputs: `{output_dir}`")

    if st.button("Run Comparison", type="primary", disabled=len(selected) < 2, key="compare_run_btn"):
        try:
            result = run_session_comparison(
                project_dir,
                selected,
                output_dir,
                good_threshold=good_threshold,
                weak_threshold=weak_threshold,
                export_heatmaps=export_heatmaps,
            )
        except Exception as e:
            st.error(str(e))
            return
        st.success("Comparison complete.")
        st.session_state["last_compare_result"] = result

    result = st.session_state.get("last_compare_result")
    if result:
        paths = project_paths(project_dir)
        st.subheader("Router locations vs sessions")
        st.caption(
            "Orange/blue ★ = labeled AP spots. Each session is one walk with the router "
            "at the matching star. Re-run comparison to refresh maps after code updates."
        )
        artifacts = result["artifacts"]
        _image_if_exists(
            artifacts.get("router_trial_map_png"),
            "Which AP location each session tested (ranked)",
        )

        routers = _load_json(paths["router_positions_json"], [])
        if routers:
            map_rows = []
            for rp in routers:
                rid = rp.get("router_position_id", "")
                name = rp.get("name", rid)
                match = next(
                    (r for r in result.get("metrics", []) if r.get("router_position_id") == rid),
                    None,
                )
                map_rows.append({
                    "AP label": name,
                    "position_id": rid,
                    "x_px": rp.get("x_px"),
                    "y_px": rp.get("y_px"),
                    "session": match.get("session_id", "—") if match else "—",
                    "rank": match.get("rank", "—") if match else "—",
                    "score": match.get("composite_score", "—") if match else "—",
                })
            st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)

        st.subheader("Ranking")
        st.dataframe(pd.DataFrame(result["metrics"]), use_container_width=True)
        _image_if_exists(artifacts.get("ranking_png"), "Ranking")
        _image_if_exists(artifacts.get("bars_png"), "Coverage / worst-case / stability")
        _image_if_exists(artifacts.get("best_vs_worst_png"), "Best vs worst (★ = best trial AP)")
        _image_if_exists(artifacts.get("walk_deltas_png"), "Matched walk deltas")
        heat_dir = artifacts.get("heatmaps_dir")
        if heat_dir and Path(heat_dir).is_dir():
            st.subheader("Per-session heatmaps")
            st.caption("Blue ★ = AP used for that session; grey/orange = other labeled candidates.")
            for png in sorted(Path(heat_dir).glob("*_heatmap.png")):
                _image_if_exists(png, png.name)
        _dataframe_if_exists(artifacts.get("metrics_csv"))
        _dataframe_if_exists(artifacts.get("walk_pairs_csv"))


def _render_optimizer(project_dir: Path, paths: dict, session_ids: list[str]):
    st.subheader("Optimize Placement")
    metadata = _load_json(paths["floorplan_metadata"], {})
    if not metadata.get("scale_pixels_per_foot"):
        st.warning("Placement optimization requires floorplan scale. Use floorplan import to measure one known wall.")
    if len(session_ids) < 3:
        st.info("Three router-position trials are recommended for better path-loss fitting.")

    selected = _session_selector(session_ids, key="optimizer_sessions", multiselect=True)
    candidate_step = st.number_input(
        "Candidate grid step (px)",
        min_value=20,
        max_value=400,
        value=80,
        step=10,
        key="optimizer_candidate_step",
    )
    receiver_step = st.number_input(
        "Receiver grid step (px)",
        min_value=20,
        max_value=400,
        value=80,
        step=10,
        key="optimizer_receiver_step",
    )
    top_k = st.number_input(
        "Top candidates",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="optimizer_top_k",
    )
    ap_height = st.number_input(
        "Suggested AP height (ft)",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.5,
        key="optimizer_ap_height",
    )
    output_dir = _project_output_dir(project_dir) / "placement"
    st.caption(f"Outputs: `{output_dir}`")

    disabled = not bool(metadata.get("scale_pixels_per_foot")) or not selected
    if st.button("Run Placement Optimizer", type="primary", disabled=disabled, key="optimizer_run_btn"):
        try:
            result = optimize_placement(
                project_dir,
                selected,
                output_dir,
                candidate_step_px=int(candidate_step),
                receiver_step_px=int(receiver_step),
                top_k=int(top_k),
                ap_height_ft=float(ap_height),
            )
        except Exception as e:
            st.error(str(e))
            return
        st.success("Placement optimization complete.")
        st.session_state["last_optimizer_result"] = result

    result = st.session_state.get("last_optimizer_result")
    if result:
        st.caption(
            "**Legend:** orange ★ = Hall / Fridge / Original (trials you measured). "
            "green ★ **#1** = model’s top new spot. purple dots = ranks #2–#5."
        )
        artifacts = result["artifacts"]
        _image_if_exists(
            artifacts.get("placement_overview_png"),
            "Trial APs + suggested placements (read this first)",
        )
        predicted = artifacts.get("predicted_coverage_png")
        if predicted:
            _image_if_exists(predicted, "Predicted coverage at green #1 (same markers)")

        rec = result.get("recommendation", {})
        candidates = rec.get("top_candidates", [])
        if candidates:
            st.subheader("Suggested coordinates")
            cand_df = pd.DataFrame(candidates)[
                ["rank", "x_px", "y_px", "score", "pct_area_good", "p10_dbm", "mean_dbm"]
            ]
            st.dataframe(cand_df, use_container_width=True, hide_index=True)

        with st.expander("Model parameters (JSON)"):
            _json_if_exists(artifacts.get("model_params_json"))
            _json_if_exists(artifacts.get("placement_recommendation_json"))


def _render_results(project_dir: Path):
    st.subheader("Generated Results")
    output_dir = _project_output_dir(project_dir)
    if not output_dir.exists():
        st.info("No project-local output folder exists yet.")
        return

    pngs = sorted(output_dir.glob("**/*.png"))
    csvs = sorted(output_dir.glob("**/*.csv"))
    jsons = sorted(output_dir.glob("**/*.json"))

    st.write(f"Found {len(pngs)} PNG, {len(csvs)} CSV, {len(jsons)} JSON output files.")
    with st.expander("Images", expanded=bool(pngs)):
        for path in pngs:
            _image_if_exists(path, str(path.relative_to(output_dir)))
    with st.expander("CSV tables", expanded=False):
        for path in csvs:
            st.markdown(f"**{path.relative_to(output_dir)}**")
            _dataframe_if_exists(path)
    with st.expander("JSON", expanded=False):
        for path in jsons:
            st.markdown(f"**{path.relative_to(output_dir)}**")
            _json_if_exists(path)


def _discover_presence_sessions(project_dir: Path) -> list[str]:
    root = project_dir / "presence_sessions"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _render_presence_occupancy(project_dir: Path):
    st.subheader("Whole-Home Occupancy")
    session_ids = _discover_presence_sessions(project_dir)
    if not session_ids:
        st.info("No `presence_sessions/*` folders found yet.")
        st.code(
            "./scripts/run_presence_tripwire.sh "
            f"--project {project_dir} --session home_occupancy --label-block vacant"
        )
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

    labels_df = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
    features_df = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()
    states_df = pd.read_csv(states_path) if states_path.exists() else pd.DataFrame()
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    errors_df = pd.read_csv(errors_path) if errors_path.exists() else pd.DataFrame()
    health_df = pd.read_csv(health_path) if health_path.exists() else pd.DataFrame()

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
        with st.expander("Evaluation JSON", expanded=False):
            st.json(evaluation)
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
    elif not features_df.empty:
        st.info("Feature rows exist, but no live occupancy states have been written yet.")

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

    if not labels_df.empty and not states_df.empty:
        st.subheader("Validation Summary")
        validation = labels_df[labels_df["is_training"].astype(str) == "0"]
        if validation.empty:
            st.info("No validation block labels found yet.")
        else:
            merged = validation.merge(states_df, on="window_id", how="left", suffixes=("_label", "_state"))
            st.dataframe(merged, use_container_width=True, hide_index=True)

    st.subheader("HP Commands")
    st.code(
        "./scripts/run_presence_tripwire.sh "
        f"--project {project_dir} --session {session_id} --interface wlan1 "
        "--label-block vacant --block-seconds 300"
    )
    st.code(
        "./scripts/run_presence_tripwire.sh "
        f"--project {project_dir} --session {session_id} --interface wlan1 "
        "--label-block occupied_still --block-seconds 300"
    )
    st.code(
        "./scripts/run_presence_tripwire.sh "
        f"--project {project_dir} --session {session_id} --interface wlan1 "
        "--monitor --occupancy-monitor"
    )
    st.code(
        "python3 mac_analysis/presence_training.py "
        f"--project {project_dir} --session {session_id}"
    )


def _render_hp_commands(project_dir: Path):
    st.subheader("Transfer and Collect on HP")
    st.write("Copy the project to the HP, collect sessions there, then sync `survey_sessions` back.")
    st.code(f"rsync -av {project_dir}/ user@hp-laptop:~/wifi-survey/{project_dir}/")
    st.code(f"python3 hp_collector/preflight.py --project {project_dir}")
    st.code(f"./scripts/run_collector.sh --project {project_dir}")
    st.code(
        "rsync -av user@hp-laptop:~/wifi-survey/"
        f"{project_dir}/survey_sessions/ {project_dir}/survey_sessions/"
    )
    st.write("For occupancy training, sync `presence_sessions` too.")
    st.code(
        "rsync -av user@hp-laptop:~/wifi-survey/"
        f"{project_dir}/presence_sessions/ {project_dir}/presence_sessions/"
    )
    st.code(
        "./scripts/run_presence_tripwire.sh "
        f"--project {project_dir} --session home_occupancy --interface wlan1 "
        "--label-block vacant --block-seconds 300"
    )
    st.code(
        "./scripts/run_presence_tripwire.sh "
        f"--project {project_dir} --session home_occupancy --interface wlan1 "
        "--monitor --occupancy-monitor"
    )
    st.caption("On the HP, use `python3 hp_collector/collector_launcher.py` for the button-style launcher.")


def main():
    args = _parse_args()
    project_dir = Path(args.project)
    paths = project_paths(project_dir)
    session_ids = discover_sessions(paths["survey_sessions_dir"])

    st.set_page_config(page_title="Wi-Fi Survey Dashboard", layout="wide")
    st.title("Wi-Fi Survey Dashboard")
    st.caption(f"Project: `{project_dir}`")

    tabs = st.tabs([
        "Overview",
        "Setup",
        "Heatmaps",
        "Compare Sessions",
        "Optimize Placement",
        "Presence/Occupancy",
        "Results",
        "HP Transfer/Collect",
    ])
    with tabs[0]:
        _render_overview(project_dir, paths, session_ids)
    with tabs[1]:
        _render_setup(project_dir)
    with tabs[2]:
        _render_heatmaps(project_dir, session_ids)
    with tabs[3]:
        _render_comparison(project_dir, session_ids)
    with tabs[4]:
        _render_optimizer(project_dir, paths, session_ids)
    with tabs[5]:
        _render_presence_occupancy(project_dir)
    with tabs[6]:
        _render_results(project_dir)
    with tabs[7]:
        _render_hp_commands(project_dir)


if __name__ == "__main__":
    main()
