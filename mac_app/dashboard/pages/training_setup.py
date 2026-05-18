"""Training → Setup page: project config + agent ping."""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from mac_app.transport.zmq_subscriber import Subscriber
from shared.project import project_paths


def _ping_agent(windows_endpoint: str, timeout_s: float = 3.0) -> tuple[bool, str]:
    """Subscribe briefly to the agent's window stream; return (saw_anything, detail)."""
    sub = Subscriber(windows_endpoint)
    try:
        t0 = time.monotonic()
        seen_window = False
        seen_health = False
        while time.monotonic() - t0 < timeout_s:
            envs = sub.poll_windows()
            for env in envs:
                if env.topic == "window":
                    seen_window = True
                if env.topic == "health":
                    seen_health = True
            if seen_window or seen_health:
                break
            time.sleep(0.05)
        if seen_window:
            return True, "agent is publishing windows ✓"
        if seen_health:
            return True, "agent published health (no windows yet)"
        return False, "no messages within timeout"
    finally:
        sub.close()


def render(project_dir: Path) -> None:
    st.header("Training → Setup")
    p = project_paths(project_dir)
    cfg_path = p["project_config"]
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            st.error(f"could not parse {cfg_path}")
    st.subheader("Project config")
    st.json(cfg, expanded=False)

    st.subheader("Agent endpoints")
    endpoint = st.text_input("windows ZMQ endpoint", value="tcp://localhost:5555")
    if st.button("Ping agent"):
        ok, detail = _ping_agent(endpoint, timeout_s=3.0)
        if ok:
            st.success(detail)
        else:
            st.error(detail)
            st.caption("start the agent with `presence-agent run --project ... --session ...`")
