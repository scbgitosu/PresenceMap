"""Startup validation that blocks ``presence-agent run``.

Each check returns ``(ok: bool, detail: str)`` and a short remediation hint.
The orchestrator at the bottom collects every result so the operator sees
*all* failures in one shot instead of fix-one-rerun-cycle-one.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from hp_agent.config import AgentConfig
from hp_agent.capture.rssi_iw import parse_iw_scan
from hp_agent.util.sudo import has_passwordless_sudo


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


# ---- individual checks ---------------------------------------------------- #


def check_cv2_importable() -> CheckResult:
    try:
        import cv2  # noqa: F401
    except Exception as exc:
        return CheckResult(
            "cv2_importable",
            False,
            f"import cv2 failed: {exc}",
            hint="pip install opencv-python-headless",
        )
    return CheckResult("cv2_importable", True, "cv2 importable")


def check_webcam(cfg: AgentConfig) -> CheckResult:
    if not cfg.webcam_enabled:
        return CheckResult("webcam", True, "disabled")
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:
        return CheckResult("webcam", False, f"cv2 unavailable: {exc}")
    candidates = [cfg.webcam_index_hint] + [i for i in (0, 1, 2, 3) if i != cfg.webcam_index_hint]
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                return CheckResult("webcam", True, f"camera index {idx} resolved")
    return CheckResult(
        "webcam",
        False,
        f"no working camera index in {candidates}",
        hint="verify the webcam is plugged in; try `presence-agent webcam-test --index 0`",
    )


def check_disk_writable(cfg: AgentConfig, min_free_mb: int = 500) -> CheckResult:
    sessions_dir = cfg.paths.get("sessions_dir", cfg.project_dir / "sessions")
    sessions_dir.mkdir(parents=True, exist_ok=True)
    probe = sessions_dir / ".presence_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return CheckResult("disk_writable", False, f"cannot write to {sessions_dir}: {exc}")
    usage = shutil.disk_usage(sessions_dir)
    free_mb = usage.free // (1024 * 1024)
    if free_mb < min_free_mb:
        return CheckResult(
            "disk_writable",
            False,
            f"{free_mb} MiB free under {sessions_dir}, need {min_free_mb} MiB",
        )
    return CheckResult("disk_writable", True, f"{free_mb} MiB free")


def _ip_link_type(interface: str) -> Optional[str]:
    """Read ``iw dev <iface> info`` (no sudo) for the iftype line."""
    try:
        r = subprocess.run(
            ["iw", "dev", interface, "info"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("type "):
            return line.split(None, 1)[1].strip()
    return None


def check_interface_present(cfg: AgentConfig) -> CheckResult:
    iftype = _ip_link_type(cfg.interface)
    if iftype is None:
        return CheckResult(
            "interface_present",
            False,
            f"{cfg.interface} not visible to iw",
            hint="confirm the AR9271 USB adapter is plugged in",
        )
    # In Stage 2 we still accept station mode (CSI lands in Stage 3 which requires monitor).
    if cfg.csi_enabled and iftype != "monitor":
        return CheckResult(
            "interface_present",
            False,
            f"{cfg.interface} type={iftype}, expected monitor",
            hint="run tools/scripts/atheros_csi_setup.sh",
        )
    return CheckResult("interface_present", True, f"{cfg.interface} type={iftype}")


def check_rfkill() -> CheckResult:
    if shutil.which("rfkill") is None:
        return CheckResult("rfkill", True, "rfkill not present, skipping")
    try:
        r = subprocess.run(["rfkill", "list", "wifi"], capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("rfkill", False, f"rfkill list failed: {exc}")
    blocked = any("blocked: yes" in line.lower() for line in r.stdout.splitlines())
    if blocked:
        return CheckResult(
            "rfkill",
            False,
            "wifi is rfkill-blocked",
            hint="run `sudo rfkill unblock wifi`",
        )
    return CheckResult("rfkill", True, "no rfkill block")


def check_sudo_iw() -> CheckResult:
    ok, detail = has_passwordless_sudo("iw")
    return CheckResult(
        "sudo_iw",
        ok,
        detail,
        hint='add `<user> ALL=(ALL) NOPASSWD: /usr/sbin/iw` to /etc/sudoers.d/presence',
    )


def check_target_ssid_visible(cfg: AgentConfig) -> CheckResult:
    if not cfg.target_ssid:
        return CheckResult("target_ssid_visible", True, "no target SSID configured")
    cmd = ["sudo", "-n", "iw", "dev", cfg.interface, "scan"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("target_ssid_visible", False, f"scan failed: {exc}")
    if r.returncode != 0:
        return CheckResult(
            "target_ssid_visible",
            False,
            f"iw scan exit {r.returncode}: {r.stderr.strip()[:120]}",
        )
    rows = parse_iw_scan(r.stdout)
    seen = [row for row in rows if row.ssid == cfg.target_ssid]
    if not seen:
        return CheckResult(
            "target_ssid_visible",
            False,
            f"SSID {cfg.target_ssid!r} not in {len(rows)} BSSes",
            hint=("ensure the router is on, broadcasting, and on the channel the agent monitors"),
        )
    return CheckResult(
        "target_ssid_visible",
        True,
        f"target {cfg.target_ssid!r} seen on {len(seen)} BSS(es)",
    )


# ---- orchestrator --------------------------------------------------------- #


def run_preflight(cfg: AgentConfig, *, skip_target: bool = False) -> Tuple[bool, List[CheckResult]]:
    checks: List[Callable[[], CheckResult]] = [
        check_cv2_importable,
        lambda: check_webcam(cfg),
        lambda: check_disk_writable(cfg),
        lambda: check_interface_present(cfg),
        check_rfkill,
        check_sudo_iw,
    ]
    if not skip_target:
        checks.append(lambda: check_target_ssid_visible(cfg))
    results = [fn() for fn in checks]
    all_ok = all(r.ok for r in results)
    return all_ok, results
