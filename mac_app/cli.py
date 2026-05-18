"""``presence-mac`` CLI.

Stage 2 surface: ``transport-tap`` only -- subscribe to a running HP agent and
print message envelopes for verification. ``dashboard`` and ``api`` come in
Stages 4 and 6.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from mac_app.transport.reconnect import Watchdog
from mac_app.transport.zmq_subscriber import Subscriber


def _cmd_transport_tap(args: argparse.Namespace) -> int:
    sub = Subscriber(args.windows, frames_connect=args.frames if args.with_frames else None)
    wd = Watchdog(stall_threshold_s=args.stall_after)
    stop = {"flag": False}

    def _sigint(*_a):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    frames_seen = 0
    windows_seen = 0
    healths_seen = 0
    print(f"tap listening on windows={args.windows}" + (f" frames={args.frames}" if args.with_frames else ""))
    while not stop["flag"]:
        envelopes = sub.poll_windows()
        for env in envelopes:
            if env.topic == "window":
                wd.mark_received()
                windows_seen += 1
                p = env.payload
                rssi = p.get("rssi", {})
                csi = p.get("csi", {})
                print(
                    f"[window#{windows_seen}] id={p.get('window_id')} phase={p.get('phase')} "
                    f"target_seen={rssi.get('target_seen')} rssi_avg={rssi.get('target_rssi_avg_dbm')} "
                    f"snr={rssi.get('snr_db')} csi_frames={csi.get('frames', 0)}"
                )
            elif env.topic == "health":
                healths_seen += 1
                p = env.payload
                print(f"[health#{healths_seen}] code={p.get('code')} detail={p.get('detail')!r} metrics={p.get('metrics')}")
            else:
                print(f"[?] topic={env.topic} payload={env.payload}")
        if args.with_frames:
            frame = sub.poll_frame()
            if frame is not None:
                frames_seen += 1
                print(f"[frame#{frames_seen}] agent={frame['agent_id']} ts_us={frame['ts_us']} jpeg_bytes={len(frame['jpeg'])}")
        if wd.is_signal_lost() and windows_seen > 0:
            print(f"[!] signal lost ({wd.age_s():.1f}s since last window)")
            time.sleep(1.0)
        else:
            time.sleep(0.05)
    sub.close()
    print(f"tap stopped. windows={windows_seen} healths={healths_seen} frames={frames_seen}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    entry = repo_root / "mac_app" / "dashboard" / "app.py"
    if not entry.exists():
        print(f"dashboard entry not found: {entry}", file=sys.stderr)
        return 2
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(entry),
        "--",
        "--project",
        str(args.project),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root))
    return subprocess.call(cmd, env=env)


def _cmd_api(_args: argparse.Namespace) -> int:
    print("api lands in Stage 6 (`uvicorn mac_app.api.server:app`)", file=sys.stderr)
    return 64


def _cmd_train(args: argparse.Namespace) -> int:
    from mac_app.train.features import DatasetSpec
    from mac_app.train.train import TrainConfig, train_session

    spec = DatasetSpec(sequence_length=args.T)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        val_fraction=args.val_fraction,
        device=args.device,
    )

    def _on_epoch(epoch, h):
        print(
            f"epoch {epoch:>3}  "
            f"train_loss={h.train_loss[-1]:.4f}  train_acc={h.train_acc[-1]:.3f}  "
            f"val_loss={h.val_loss[-1]:.4f}  val_acc={h.val_acc[-1]:.3f}"
        )

    try:
        meta = train_session(args.project, args.session, spec=spec, cfg=cfg, on_epoch=_on_epoch)
    except Exception as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1
    print("=" * 60)
    print(f"saved model: {meta.model_id}")
    s = meta.eval_summary
    print(
        f"acc={s['accuracy']:.3f} prec={s['precision']:.3f} "
        f"recall={s['recall']:.3f} f1={s['f1']:.3f}"
    )
    print(f"confusion (rows=true, cols=pred): {s['confusion']}")
    print(f"artifacts: {meta.model_dir}")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="presence-mac")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("transport-tap", help="subscribe to a running agent and print envelopes")
    p.add_argument("--windows", default="tcp://localhost:5555", help="ZMQ connect address for window+health stream")
    p.add_argument("--frames", default="tcp://localhost:5556", help="ZMQ connect address for frames stream")
    p.add_argument("--with-frames", action="store_true", help="also subscribe to frames")
    p.add_argument("--stall-after", type=float, default=6.0, help="seconds before declaring signal lost")
    p.set_defaults(func=_cmd_transport_tap)

    p = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    p.add_argument("--project", default="data/survey_projects/apartment_test")
    p.set_defaults(func=_cmd_dashboard)

    p = sub.add_parser("api", help="(Stage 6) launch the FastAPI live state service")
    p.set_defaults(func=_cmd_api)

    p = sub.add_parser("train", help="train a TemporalCSIModel on one or more recorded sessions")
    p.add_argument("--project", required=True, help="path to project directory")
    p.add_argument("--session", required=True, nargs="+", help="one or more v2 session ids")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.3)
    p.add_argument("-T", "--T", dest="T", type=int, default=8, help="sequence length")
    p.add_argument("--device", default=None, help="mps | cpu | cuda (default: mps if available)")
    p.set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
