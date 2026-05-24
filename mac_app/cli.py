"""``presence-mac`` CLI."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_project(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    return p


def _cmd_dashboard(args: argparse.Namespace) -> int:
    entry = _repo_root() / "mac_app" / "dashboard" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(entry),
        "--",
        "--project",
        str(_resolve_project(args.project)),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_repo_root()))
    return subprocess.call(cmd, env=env)


def _cmd_api(args: argparse.Namespace) -> int:
    if args.sqlite:
        os.environ["PRESENCE_SQLITE_PATH"] = str(args.sqlite)
    if args.project:
        os.environ["PRESENCE_PROJECT_DIR"] = str(_resolve_project(args.project))
    if args.bind:
        os.environ["PRESENCE_API_BIND"] = args.bind
    from mac_app.api.server import main as api_main

    return api_main()


def _cmd_esp32_ingest(args: argparse.Namespace) -> int:
    from mac_app.capture.esp32_ingest import Esp32IngestService

    repo_root = _repo_root()
    svc = Esp32IngestService(
        repo_root,
        udp_port=args.udp_port,
        window_seconds=args.window_seconds,
    )
    print(f"ESP32 ingest UDP :{args.udp_port}")
    print(f"State: {svc.store.path}")
    print("Ctrl+C to stop")
    svc.run_forever()
    return 0


def _cmd_esp32_record(args: argparse.Namespace) -> int:
    from mac_app.capture.esp32_recorder import record_session

    try:
        out = record_session(
            _repo_root(),
            _resolve_project(args.project),
            args.session,
            duration_s=args.duration,
            udp_port=args.udp_port,
            phase=args.phase,
        )
    except Exception as exc:
        print(f"record failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


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
            f"epoch {epoch:>3}  train_loss={h.train_loss[-1]:.4f}  "
            f"train_acc={h.train_acc[-1]:.3f}  val_acc={h.val_acc[-1]:.3f}"
        )

    try:
        meta = train_session(
            _resolve_project(args.project),
            args.session,
            spec=spec,
            cfg=cfg,
            on_epoch=_on_epoch,
        )
    except Exception as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1
    print(f"saved {meta.model_id}")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="presence-mac")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dashboard", help="launch Streamlit UI")
    p.add_argument("--project", default="data/survey_projects/my_bed")
    p.set_defaults(func=_cmd_dashboard)

    p = sub.add_parser("api", help="local REST API for live state")
    p.add_argument("--project", default="data/survey_projects/my_bed")
    p.add_argument("--sqlite", default=None)
    p.add_argument("--bind", default=None)
    p.set_defaults(func=_cmd_api)

    p = sub.add_parser("esp32-ingest", help="UDP CSI ingest + diagnostics state file")
    p.add_argument("--project", default="data/survey_projects/my_bed")
    p.add_argument("--udp-port", type=int, default=5005)
    p.add_argument("--window-seconds", type=float, default=2.0)
    p.set_defaults(func=_cmd_esp32_ingest)

    p = sub.add_parser("esp32-record", help="record ESP32 windows to session parquet")
    p.add_argument("--project", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--udp-port", type=int, default=5005)
    p.add_argument("--phase", default="labeled_vacant")
    p.set_defaults(func=_cmd_esp32_record)

    p = sub.add_parser("train", help="train TemporalCSIModel on ESP32 sessions")
    p.add_argument("--project", required=True)
    p.add_argument("--session", required=True, nargs="+")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.3)
    p.add_argument("-T", dest="T", type=int, default=8)
    p.add_argument("--device", default=None)
    p.set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
