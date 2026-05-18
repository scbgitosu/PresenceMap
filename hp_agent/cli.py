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

from hp_agent.config import AgentConfig
from hp_agent.health.heartbeat import HeartbeatEmitter
from hp_agent.health.preflight import run_preflight
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


def _cmd_csi_test(args: argparse.Namespace) -> int:
    print(
        "csi-test not wired yet -- lands in Stage 3 once recvCSI integration "
        "and ath9k monitor mode are in place.",
        file=sys.stderr,
    )
    return 64


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = AgentConfig.from_project(
        args.project,
        agent_id=args.agent_id,
        overrides={"interface": args.interface},
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
    win = WindowLoop(cfg, publisher, heartbeat, session_id=session_id, phase=args.phase)
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

    p = sub.add_parser("csi-test", help="(Stage 3) capture a few CSI windows")
    p.add_argument("--iface", default="wlan1")
    p.add_argument("--duration", type=float, default=5.0)
    p.set_defaults(func=_cmd_csi_test)

    p = sub.add_parser("run", help="run the main agent loop")
    _add_common_args(p)
    p.add_argument("--session", required=True, help="session id (used as directory name)")
    p.add_argument("--phase", default="freeform", choices=["calibration", "labeled_vacant", "labeled_occupied", "freeform", "live"])
    p.add_argument("--no-webcam", action="store_true")
    p.add_argument("--no-preflight", action="store_true")
    p.add_argument("--skip-target", action="store_true")
    p.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
