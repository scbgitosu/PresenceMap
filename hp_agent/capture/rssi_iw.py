"""``iw`` scan + per-BSS signal extraction + noise-floor reading.

Replaces the v1 ``hp_collector/wifi_scan.py`` link-based SNR path, which
silently failed in monitor mode (``iw link`` returns "Not connected" because
the interface is never associated). Here we:

1. Parse per-BSS ``signal_dbm`` directly from ``iw dev <iface> scan`` output.
   This works in both station mode and (with the AR9271/ath9k patched driver)
   monitor mode, because the kernel injects rx signal levels into beacon BSS
   reports.
2. Read the noise floor from ``/sys/kernel/debug/ieee80211/phy*/ath9k/dump_nf``
   when available, falling back to ``iw dev <iface> survey dump | grep noise``.
3. Compute ``snr_db = signal_dbm - noise_dbm`` only when *both* are populated;
   otherwise we leave ``snr_db = None`` and emit a structured warning.

If a scan subprocess fails, the caller (``window_loop``) is responsible for
emitting a HealthMsg with the structured error. We do not swallow it here.
"""
from __future__ import annotations

import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class BssRow:
    bssid: str
    ssid: str
    signal_dbm: Optional[float]
    frequency_mhz: Optional[float]
    channel: Optional[int]


@dataclass
class RFSummary:
    target_rssi_avg_dbm: Optional[float] = None
    target_rssi_std_db: Optional[float] = None
    snr_db: Optional[float] = None
    noise_dbm: Optional[float] = None
    visible_bssid_count: int = 0
    neighbor_rssi_sum_dbm: Optional[float] = None
    channel_utilization_proxy: Optional[float] = None
    target_seen: bool = False
    # Raw rows preserved for diagnostics / future per-AP features
    bss_rows: List[BssRow] = field(default_factory=list)


_BSS_RE = re.compile(r"^BSS\s+([0-9a-fA-F:]{17})", re.MULTILINE)
_FREQ_RE = re.compile(r"^\s*freq:\s*(\d+)", re.MULTILINE)
_SIGNAL_RE = re.compile(r"^\s*signal:\s*(-?\d+\.\d+)\s*dBm", re.MULTILINE)
_SSID_RE = re.compile(r"^\s*SSID:\s*(.*)$", re.MULTILINE)


def _channel_from_freq(freq_mhz: int) -> Optional[int]:
    if freq_mhz is None:
        return None
    if 2412 <= freq_mhz <= 2484:
        if freq_mhz == 2484:
            return 14
        return 1 + (freq_mhz - 2412) // 5
    if 5170 <= freq_mhz <= 5825:
        return (freq_mhz - 5000) // 5
    return None


def parse_iw_scan(text: str) -> List[BssRow]:
    """Parse the output of ``iw dev <iface> scan``."""
    rows: List[BssRow] = []
    # Split per-BSS blocks. Use the BSS line as a delimiter and keep the trailing block.
    blocks = re.split(r"(?m)^BSS\s+", text)
    if not blocks:
        return rows
    for block in blocks[1:]:
        bss = block[:17]
        rest = block[17:]
        freq_m = _FREQ_RE.search(rest)
        sig_m = _SIGNAL_RE.search(rest)
        ssid_m = _SSID_RE.search(rest)
        freq = int(freq_m.group(1)) if freq_m else None
        signal = float(sig_m.group(1)) if sig_m else None
        ssid = ssid_m.group(1).strip() if ssid_m else ""
        rows.append(
            BssRow(
                bssid=bss.lower(),
                ssid=ssid,
                signal_dbm=signal,
                frequency_mhz=float(freq) if freq is not None else None,
                channel=_channel_from_freq(freq) if freq is not None else None,
            )
        )
    return rows


def run_iw_scan(interface: str, *, timeout_s: float = 6.0, use_sudo: bool = True) -> List[BssRow]:
    """Run ``iw dev <iface> scan`` and return parsed BSS rows.

    Raises ``subprocess.CalledProcessError`` on non-zero exit so the caller
    can route it into a HealthMsg (and not swallow it like v1 did).
    """
    cmd = (["sudo", "-n"] if use_sudo else []) + ["iw", "dev", interface, "scan"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise subprocess.CalledProcessError(
            r.returncode, cmd, output=r.stdout, stderr=r.stderr
        )
    return parse_iw_scan(r.stdout)


_NOISE_DBM_RE = re.compile(r"noise:\s*(-?\d+)\s*dBm")


def read_noise_floor_dbm(interface: str = "wlan1") -> Optional[float]:
    """Best-effort noise floor in dBm. Returns ``None`` when unavailable."""
    # Prefer ath9k debugfs (gives a sane, recent NF).
    for phy in Path("/sys/kernel/debug/ieee80211").glob("phy*"):
        nf_path = phy / "ath9k" / "dump_nf"
        if nf_path.exists():
            try:
                text = nf_path.read_text()
                # File format: lines like "Channel Noise Floor : -95 dBm"
                m = re.search(r"(-?\d+)\s*dBm", text)
                if m:
                    return float(m.group(1))
            except OSError:
                continue
    # Fall back to ``iw survey dump``.
    try:
        r = subprocess.run(
            ["sudo", "-n", "iw", "dev", interface, "survey", "dump"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode != 0:
            return None
        # Prefer the in-use frequency block.
        for block in r.stdout.split("Survey data from "):
            if "in use" in block:
                m = _NOISE_DBM_RE.search(block)
                if m:
                    return float(m.group(1))
        m = _NOISE_DBM_RE.search(r.stdout)
        if m:
            return float(m.group(1))
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def summarize_window(
    rows: List[BssRow],
    *,
    target_ssid: str,
    target_bssid: str = "",
    noise_dbm: Optional[float] = None,
) -> RFSummary:
    """Aggregate raw BSS rows from one or more scans into an ``RFSummary``."""
    if not rows:
        return RFSummary(noise_dbm=noise_dbm)

    target_signals: List[float] = []
    neighbors_dbm: List[float] = []
    for r in rows:
        if r.signal_dbm is None:
            continue
        is_target = (target_bssid and r.bssid == target_bssid.lower()) or (
            target_ssid and r.ssid == target_ssid
        )
        if is_target:
            target_signals.append(r.signal_dbm)
        else:
            neighbors_dbm.append(r.signal_dbm)

    target_avg: Optional[float] = (
        statistics.fmean(target_signals) if target_signals else None
    )
    target_std: Optional[float] = (
        statistics.pstdev(target_signals) if len(target_signals) >= 2 else None
    )
    neighbor_sum_dbm: Optional[float] = (
        # Linear sum in mW then back to dBm — appropriate for "interference"
        _sum_dbm(neighbors_dbm) if neighbors_dbm else None
    )
    snr = None
    if target_avg is not None and noise_dbm is not None:
        snr = target_avg - noise_dbm

    return RFSummary(
        target_rssi_avg_dbm=target_avg,
        target_rssi_std_db=target_std,
        snr_db=snr,
        noise_dbm=noise_dbm,
        visible_bssid_count=len({r.bssid for r in rows}),
        neighbor_rssi_sum_dbm=neighbor_sum_dbm,
        channel_utilization_proxy=None,  # filled by capture/csi_features in Stage 3
        target_seen=bool(target_signals),
        bss_rows=rows,
    )


def _sum_dbm(values: Iterable[float]) -> float:
    """Power-domain sum of dBm values, expressed as dBm."""
    total_mw = 0.0
    for v in values:
        total_mw += 10.0 ** (v / 10.0)
    if total_mw <= 0:
        return float("-inf")
    import math

    return 10.0 * math.log10(total_mw)
