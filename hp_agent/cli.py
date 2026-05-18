"""``presence-agent`` CLI: preflight | csi-test | webcam-test | run.

Stage 2 surface: ``preflight``, ``webcam-test``, and ``run`` (without CSI).
``csi-test`` is a stub that prints an explanatory error until Stage 3 wires
it up.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from hp_agent.capture.csi_atheros import (
    AtherosCSIReader,
    CSIFrame,
    FixtureCSISource,
)
from hp_agent.capture.csi_features import (
    DEFAULT_NUM_SUBCARRIERS,
    aggregate_window_features,
    csi_feature_names,
)
from hp_agent.config import AgentConfig
from hp_agent.health.heartbeat import HeartbeatEmitter
from hp_agent.health.preflight import run_preflight
from hp_agent.runners.csi_buffer import CSIBuffer, CSICaptureThread
from hp_agent.runners.frame_loop import FrameLoop
from hp_agent.runners.window_loop import WindowLoop
from hp_agent.transport.zmq_publisher import Publisher
from hp_agent.util.logging import get_logger, log_event


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", required=True, help="path to project directory (data/survey_projects/<name>)")
    p.add_argument("--agent-id", default=None, help="override agent_id (default: hostname)")
    p.add_argument("--interface", default=None, help="override RF interface (default: from project_config.json)")


def _cmd_preflight(args: argparse.Namespace) -> int:
    cfg = AgentConfig.from_project(
        args.project,
        agent_id=args.agent_id,
        overrides={"interface": args.interface},
    )
    ok, results = run_preflight(cfg, skip_target=args.skip_target)
    width = max(len(r.name) for r in results)
    for r in results:
        marker = "PASS" if r.ok else "FAIL"
        print(f"  {marker}  {r.name.ljust(width)}  {r.detail}")
        if not r.ok and r.hint:
            print(f"        hint: {r.hint}")
    return 0 if ok else 1


def _cmd_webcam_test(args: argparse.Namespace) -> int:
    import cv2  # type: ignore[import-not-found]

    from hp_agent.capture.webcam_stream import resolve_camera_index

    idx = resolve_camera_index(args.index)
    if idx is None:
        print(f"no working camera in [{args.index}, 0, 1, 2, 3]", file=sys.stderr)
        return 2
    print(f"resolved camera index: {idx}")
    cap = cv2.VideoCapture(idx)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    while written < args.frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("read returned no frame", file=sys.stderr)
            break
        path = out_dir / f"frame_{written:03d}.jpg"
        cv2.imwrite(str(path), frame)
        written += 1
        time.sleep(0.1)
    cap.release()
    print(f"wrote {written} frames to {out_dir}")
    return 0 if written == args.frames else 3


def _make_fixture_source(num_subcarriers: int, rate_hz: float = 30.0) -> FixtureCSISource:
    """Synthesize a tiny ring of CSI frames -- useful for dev on macOS."""
    rng = np.random.default_rng(seed=0)
    frames = []
    for i in range(10):
        amps = np.full(num_subcarriers, 100.0) + rng.normal(0, 1.0, num_subcarriers).astype(np.float32)
        csi = np.zeros((1, 1, num_subcarriers), dtype=np.complex64)
        csi[0, 0, :] = amps
        frames.append(
            CSIFrame(
                ts=0.0, csi=csi,
                nr=1, nc=1, num_tones=num_subcarriers,
                bandwidth=0,
                rssi=200, rssi_1=200, rssi_2=0, rssi_3=0,
                noise_dbm=-95, rate=7, tstamp_ns=i,
            )
        )
    return FixtureCSISource(frames, rate_hz=rate_hz)


def _make_csi_source(cfg: AgentConfig, *, override: Optional[str] = None):
    """Return a CSI source matching ``cfg.csi_source`` (or the override)."""
    src = override or cfg.csi_source
    if src == "fixture":
        return _make_fixture_source(cfg.csi_num_subcarriers)
    if src == "udp":
        return AtherosCSIReader(udp_port=cfg.csi_recv_port)
    if src == "atheros":
        return AtherosCSIReader(
            binary=cfg.csi_recv_binary,
            binary_args=cfg.csi_recv_binary_args,
        )
    raise ValueError(f"unknown csi source {src!r} (expected fixture|udp|atheros)")


def _cmd_csi_test(args: argparse.Namespace) -> int:
    cfg = AgentConfig.from_project(args.project, overrides={"interface": args.iface})
    cfg.csi_enabled = True
    cfg.csi_source = args.source
    buffer = CSIBuffer()
    source = _make_csi_source(cfg, override=args.source)
    thread = CSICaptureThread(source, buffer)
    thread.start()
    print(f"capturing {args.duration}s of CSI from {args.source}...")
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.duration:
            time.sleep(0.1)
    finally:
        thread.stop()
    frames = buffer.drain()
    print(f"captured {len(frames)} frames")
    if not frames:
        print("no frames -- check setup (kernel patch, monitor mode, recvCSI binary)", file=sys.stderr)
        return 2
    feats, amp_blob, loss = aggregate_window_features(
        frames,
        expected_frames=int(args.duration * 30),
        num_subcarriers=cfg.csi_num_subcarriers,
        include_amp_matrix=False,
    )
    names = csi_feature_names(cfg.csi_num_subcarriers)
    K = cfg.csi_num_subcarriers
    print(
        f"loss_ratio={loss:.3f}  "
        f"amp_mean_avg={sum(feats[:K])/K:.3f}  "
        f"amp_std_avg={sum(feats[K:2*K])/K:.3f}  "
        f"phase_std_avg={sum(feats[2*K:3*K])/K:.3f}"
    )
    print(
        f"amp_total_var={feats[3*K]:.4f}  "
        f"pca_top1_eig={feats[3*K+1]:.4f}  "
        f"pca_top1_ratio={feats[3*K+2]:.4f}  "
        f"subc_corr={feats[3*K+3]:.4f}  "
        f"acf1={feats[3*K+4]:.4f}"
    )
    rssi_a = sum(f.rssi for f in frames) / len(frames)
    noise = sum(f.noise_dbm for f in frames) / len(frames)
    print(f"mean rssi={rssi_a:.1f}  mean noise={noise:.1f} dBm")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    overrides = {"interface": args.interface}
    if getattr(args, "csi", False):
        overrides["csi_enabled"] = True
    if getattr(args, "csi_source", None):
        overrides["csi_source"] = args.csi_source
    cfg = AgentConfig.from_project(
        args.project,
        agent_id=args.agent_id,
        overrides=overrides,
    )
    if not args.no_preflight:
        ok, results = run_preflight(cfg, skip_target=args.skip_target)
        if not ok:
            print("preflight failed:", file=sys.stderr)
            for r in results:
                if not r.ok:
                    print(f"  - {r.name}: {r.detail}", file=sys.stderr)
                    if r.hint:
                        print(f"    hint: {r.hint}", file=sys.stderr)
            return 1
    session_id = args.session
    sessions_dir = cfg.paths.get("sessions_dir", cfg.project_dir / "sessions")
    sess_dir = sessions_dir / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("hp_agent", session_log=sess_dir / "agent.log.jsonl")
    log_event(
        log, "INFO", "agent starting",
        agent_id=cfg.agent_id, interface=cfg.interface, session_id=session_id,
        publish_bind=cfg.publish_bind, frames_bind=cfg.frames_bind,
    )
    publisher = Publisher(cfg.publish_bind, cfg.frames_bind)
    heartbeat = HeartbeatEmitter(publisher, agent_id=cfg.agent_id, session_id=session_id)
    csi_buffer: Optional[CSIBuffer] = None
    csi_thread: Optional[CSICaptureThread] = None
    if cfg.csi_enabled:
        csi_buffer = CSIBuffer()
        csi_thread = CSICaptureThread(_make_csi_source(cfg), csi_buffer)
        csi_thread.start()
        log_event(log, "INFO", "csi capture started", source=cfg.csi_source, subcarriers=cfg.csi_num_subcarriers)
    win = WindowLoop(
        cfg, publisher, heartbeat,
        session_id=session_id, phase=args.phase,
        csi_buffer=csi_buffer,
    )
    fl: Optional[FrameLoop] = None
    if not args.no_webcam:
        fl = FrameLoop(cfg, publisher, heartbeat, session_id=session_id)
        fl.start()

    def _sigint(_signum, _frame):
        log_event(log, "INFO", "received SIGINT, stopping")
        win.stop()
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)
    try:
        win.run()
    finally:
        if fl is not None:
            fl.stop()
        if csi_thread is not None:
            csi_thread.stop()
        publisher.close()
        log_event(log, "INFO", "agent stopped")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="presence-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight", help="run startup validation")
    _add_common_args(p)
    p.add_argument("--skip-target", action="store_true", help="skip target-SSID scan check")
    p.set_defaults(func=_cmd_preflight)

    p = sub.add_parser("webcam-test", help="capture N frames to a directory")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--out", default="/tmp/presence_webcam_test")
    p.set_defaults(func=_cmd_webcam_test)

    p = sub.add_parser("csi-test", help="capture CSI for N seconds and print per-window features")
    p.add_argument("--project", required=True, help="path to project directory")
    p.add_argument("--iface", default=None, help="override RF interface")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument(
        "--source",
        default="atheros",
        choices=("atheros", "udp", "fixture"),
        help="atheros=spawn recvCSI binary, udp=read UDP packets from recvCSI, fixture=synthesized frames for dev",
    )
    p.set_defaults(func=_cmd_csi_test)

    p = sub.add_parser("run", help="run the main agent loop")
    _add_common_args(p)
    p.add_argument("--session", required=True, help="session id (used as directory name)")
    p.add_argument("--phase", default="freeform", choices=["calibration", "labeled_vacant", "labeled_occupied", "freeform", "live"])
    p.add_argument("--no-webcam", action="store_true")
    p.add_argument("--no-preflight", action="store_true")
    p.add_argument("--skip-target", action="store_true")
    p.add_argument("--csi", action="store_true", help="enable CSI capture (overrides project_config.json)")
    p.add_argument("--csi-source", default=None, choices=("atheros", "udp", "fixture"))
    p.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
