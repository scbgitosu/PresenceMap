"""
Wi-Fi preflight checks before field surveying.

CLI:
    python3 hp_collector/preflight.py --project survey_projects/apartment_test
    python3 -m hp_collector.preflight --project survey_projects/apartment_test
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hp_collector.config_loader import load_project
from hp_collector.wifi_scan import interface_operstate, list_wifi_interfaces, scan


@dataclass
class PreflightResult:
    ok: bool
    issues: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)

    def message(self) -> str:
        lines = []
        if self.ok:
            lines.append("Wi-Fi preflight passed.")
        else:
            lines.append("Wi-Fi preflight failed:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
            if self.hints:
                lines.append("")
                lines.append("Try:")
                for hint in self.hints:
                    lines.append(f"  {hint}")
        return "\n".join(lines)


def check_rfkill_blocked() -> List[str]:
    """Return human-readable descriptions of rfkill blocks affecting Wi-Fi."""
    blocked = []
    try:
        out = subprocess.check_output(
            ["rfkill", "list"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return blocked

    blocks = re.split(r"\n(?=\d+:\s)", out.strip())
    for block in blocks:
        if not re.search(r"wlan|wifi|Wireless", block, re.IGNORECASE):
            continue
        if "Soft blocked: yes" not in block and "Hard blocked: yes" not in block:
            continue
        first_line = block.splitlines()[0].strip() if block.splitlines() else "rfkill block"
        blocked.append(first_line)

    return blocked


def run_preflight(
    interface: str,
    ssid: str,
    bssid: Optional[str] = None,
    backend: str = "iw",
) -> PreflightResult:
    """Run all preflight checks. Returns ok=False if surveying should be blocked."""
    issues: List[str] = []
    hints: List[str] = []

    if not ssid.strip():
        issues.append("Target SSID is empty.")
        hints.append("Set the target SSID in project config or the collector sidebar.")
        return PreflightResult(ok=False, issues=issues, hints=hints)

    ifaces = list_wifi_interfaces()
    if interface not in ifaces:
        detected = ", ".join(ifaces) if ifaces else "(none detected)"
        issues.append(f"Interface '{interface}' not found. Detected: {detected}")
        hints.append("Plug in the USB Wi-Fi adapter and wait a few seconds.")
        hints.append("Run: nmcli device  or  iw dev")
        hints.append("Update default_interface in project_config.json if the name changed.")
        return PreflightResult(ok=False, issues=issues, hints=hints)

    # operstate DOWN + NO-CARRIER is normal on scan-only Wi-Fi adapters that are
    # not associated with an AP; the trial scan below is the real readiness test.
    operstate = interface_operstate(interface)
    if operstate and operstate.upper() != "UP":
        hints.append(
            f"Note: {interface} reports operstate {operstate} "
            "(common when not connected; scanning may still work)."
        )

    rfkill_blocks = check_rfkill_blocked()
    for block in rfkill_blocks:
        issues.append(block)
    if rfkill_blocks:
        hints.append("sudo rfkill unblock wifi")

    if issues:
        return PreflightResult(ok=False, issues=issues, hints=_dedupe(hints))

    try:
        aps, backend_used = scan(interface, ssid.strip(), bssid or None, backend=backend)
    except RuntimeError as e:
        issues.append(f"Scan failed ({e}).")
        hints.append(f"sudo ip link set {interface} up")
        hints.append("sudo rfkill unblock wifi")
        if backend in ("iw", "auto"):
            hints.append(f"sudo iw dev {interface} scan >/dev/null")
            hints.append(
                "For unattended surveys, configure passwordless sudo for the iw binary."
            )
        hints.append(f"python3 hp_collector/wifi_scan.py --interface {interface} --ssid \"{ssid}\" --samples 3 --backend {backend}")
        return PreflightResult(ok=False, issues=issues, hints=_dedupe(hints))

    if not aps:
        issues.append(f"Scan succeeded via {backend_used} but target SSID '{ssid}' was not found.")
        hints.append("Confirm the router is on and broadcasting the expected SSID.")
        hints.append("Check 2.4 vs 5 GHz — the adapter may not see all bands.")
        if bssid:
            hints.append("Clear target BSSID if roaming between APs with the same SSID.")
        return PreflightResult(ok=False, issues=issues, hints=hints)

    return PreflightResult(ok=True)


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Wi-Fi survey preflight checks")
    parser.add_argument("--project", required=True, help="Path to project directory")
    parser.add_argument("--interface", default=None, help="Override Wi-Fi interface")
    parser.add_argument("--ssid", default=None, help="Override target SSID")
    parser.add_argument("--bssid", default=None, help="Override target BSSID")
    parser.add_argument("--backend", choices=["iw", "auto", "nmcli"], default=None,
                        help="Override scan backend")
    args = parser.parse_args()

    project_dir = Path(args.project)
    config, _, _, _ = load_project(project_dir, require_spatial=False)

    interface = args.interface or config.default_interface
    ssid = args.ssid or config.target_ssid
    bssid = args.bssid if args.bssid is not None else (config.target_bssid or None)
    backend = args.backend or getattr(config, "scan_backend", "iw")

    if not interface:
        print("ERROR: No Wi-Fi interface configured.", file=sys.stderr)
        print("Set default_interface in project_config.json or pass --interface.", file=sys.stderr)
        return 1

    result = run_preflight(interface, ssid, bssid or None, backend=backend)
    print(result.message())
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
