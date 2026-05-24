"""Live → ESP32 Nodes diagnostics dashboard."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from mac_app.capture.esp32_recorder import record_session
from mac_app.capture.esp32_state import Esp32RuntimeState, Esp32StateStore
from mac_app.inference.esp32_runtime import Esp32InferenceConfig, Esp32InferenceLoop
from mac_app.train.registry import list_models


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_state(repo_root: Path) -> Esp32RuntimeState | None:
    return Esp32StateStore.load(repo_root)


def _status_emoji(status: str) -> str:
    return {"ok": "🟢", "stale": "🟡", "offline": "🔴"}.get(status, "⚪")


def render(project_dir: Path) -> None:
    st.header("Live → ESP32 Nodes")
    st.info(
        "**Diagnostics only** — no pose or clinical vitals. "
        "Run `presence-mac esp32-ingest` in a terminal (see [ESP32_SETUP.md](../../../docs/ESP32_SETUP.md))."
    )

    repo = _repo_root()
    state = _load_state(repo)
    if state is None:
        st.warning("No `data/runtime/esp32_state.json` yet. Start ingest:")
        st.code("presence-mac esp32-ingest --project " + str(project_dir), language="bash")
        return

    agg = state.aggregate or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total PPS", f"{agg.get('total_pps', 0):.1f}")
    c2.metric("Nodes OK", agg.get("nodes_ok", 0))
    c3.metric("CSI frames", agg.get("csi_frames_parsed", 0))
    c4.metric("Non-CSI skipped", agg.get("sibling_packets_skipped", 0))
    c5.metric("Parse errors", agg.get("parse_errors", 0))
    st.caption(
        f"Ingest {'running' if state.ingest_running else 'stopped'} · "
        f"UDP :{state.udp_port} · window {state.window_seconds}s · "
        f"updated {state.updated_at}"
    )

    if not state.ingest_running:
        st.warning(
            "Ingest is **stopped**. Start it in another terminal (dashboard does not bind UDP):"
        )
        st.code("presence-mac esp32-ingest --project " + str(project_dir), language="bash")
    elif not state.nodes:
        skipped = int(agg.get("sibling_packets_skipped", 0))
        csi = int(agg.get("csi_frames_parsed", 0))
        if skipped > 0 and csi == 0:
            st.error(
                "UDP is arriving but **no raw CSI** (`0xC5110001`) was parsed. "
                "Your tcpdump lengths (~60 B) look like RuView **vitals/feature** packets, "
                "not ADR-018 CSI (~150+ B). Reflash or re-provision nodes to stream raw CSI "
                "(see [ESP32_SETUP.md](../../../docs/ESP32_SETUP.md) troubleshooting)."
            )
        else:
            st.warning(
                "Ingest is running but no nodes yet. Confirm node `target-ip` is this Mac "
                "and check tcpdump on UDP :5005."
            )
    else:
        for node in state.nodes:
            with st.container(border=True):
                cols = st.columns([1, 2, 2, 2, 2])
                cols[0].markdown(f"### {_status_emoji(node.get('status', 'offline'))} Node {node.get('node_id')}")
                cols[1].write(f"**IP** `{node.get('source_ip')}`")
                cols[2].metric("PPS (1s)", f"{node.get('pps_1s', 0):.1f}")
                cols[3].metric("Seq gaps", f"{100 * node.get('seq_gap_ratio', 0):.1f}%")
                cols[4].metric("Last seen", f"{node.get('last_seen_s_ago', -1):.1f}s")
                st.caption(
                    f"RSSI {node.get('rssi_dbm')} dBm · "
                    f"{node.get('n_subcarriers')} subcarriers · "
                    f"{node.get('frames_last_window')} frames/window · "
                    f"loss {node.get('loss_ratio_last_window', 0):.2%}"
                )

        hist = state.pps_history or {}
        if hist:
            st.subheader("PPS (60s)")
            frames = []
            for key, points in hist.items():
                for pt in points:
                    frames.append({"node": key, "t": pt["t"], "pps": pt["pps"]})
            if frames:
                df = pd.DataFrame(frames)
                df["t"] = pd.to_datetime(df["t"], unit="s")
                pivot = df.pivot_table(index="t", columns="node", values="pps", aggfunc="last")
                # Altair treats ":" in column names as encoding shorthand (e.g. "192.168.1.1:2" → type "2").
                pivot = pivot.rename(columns=lambda c: str(c).replace(":", " · "))
                st.line_chart(pivot)

    if state.recent_log:
        with st.expander("Ingest log"):
            st.code("\n".join(state.recent_log[-20:]))

    st.divider()
    st.subheader("Record session")
    rec_cols = st.columns([2, 2, 2, 1])
    session_id = rec_cols[0].text_input("session id", value="esp32_bed_001")
    phase = rec_cols[1].selectbox(
        "phase",
        ["calibration", "labeled_vacant", "labeled_occupied", "labeled_occupied_moving"],
    )
    duration = rec_cols[2].number_input("seconds", min_value=10, max_value=3600, value=60)
    if rec_cols[3].button("Record", type="primary"):
        if not state.ingest_running:
            st.error("Start `presence-mac esp32-ingest` first, or recording will bind UDP alone.")
        with st.spinner(f"Recording {duration}s…"):
            try:
                out = record_session(
                    repo,
                    project_dir,
                    session_id,
                    duration_s=float(duration),
                    phase=phase,
                )
                st.success(f"Wrote {out}")
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("ESP32 live inference (Phase 1)")
    models = list_models(project_dir)
    if not models:
        st.caption("Train a model on ESP32 sessions first (Training → Train).")
    else:
        loop_key = "esp32_inference_loop"
        loop = st.session_state.get(loop_key)
        model_id = st.selectbox("model", [m.model_id for m in models])
        inf_cols = st.columns(2)
        if loop is None and inf_cols[0].button("Start ESP32 live"):
            try:
                cfg = Esp32InferenceConfig(
                    project_dir=project_dir,
                    model_id=model_id,
                    repo_root=repo,
                    device="mps",
                )
                loop = Esp32InferenceLoop(cfg)
                loop.start()
                st.session_state[loop_key] = loop
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if loop is not None and inf_cols[1].button("Stop ESP32 live"):
            loop.stop()
            st.session_state.pop(loop_key, None)
            st.rerun()

    auto = st.checkbox("Auto refresh (1.5s)", value=True)
    if auto:
        time.sleep(1.5)
        st.rerun()
