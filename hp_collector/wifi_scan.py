"""
Wi-Fi scanning backend: iw (primary), nmcli (fallback).

CLI usage:
    python3 hp_collector/wifi_scan.py --interface wlan1 --ssid "MySSID" --samples 10
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import logging
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from shared.models import Sample
from shared.utils import now_iso

logger = logging.getLogger(__name__)


def list_wifi_interfaces() -> List[str]:
    """Return Wi-Fi interface names visible to nmcli or iw."""
    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "DEVICE,TYPE", "device"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        ifaces = []
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and parts[1].strip() == "wifi":
                ifaces.append(parts[0].strip())
        if ifaces:
            return ifaces
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        out = subprocess.check_output(["iw", "dev"], text=True, stderr=subprocess.DEVNULL)
        ifaces = re.findall(r"Interface\s+(\S+)", out)
        if ifaces:
            return ifaces
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return []


def interface_operstate(interface: str) -> Optional[str]:
    """Return kernel operstate for interface (e.g. UP, DOWN), or None if unknown."""
    try:
        out = subprocess.check_output(
            ["ip", "-j", "link", "show", interface],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        data = json.loads(out)
        if data and isinstance(data, list):
            return data[0].get("operstate")
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError, IndexError):
        pass

    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "DEVICE,STATE", "device", "status"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and parts[0].strip() == interface:
                state = parts[1].strip().lower()
                if state in ("connected", "disconnected"):
                    return "UP"
                if state == "unavailable":
                    return "DOWN"
                return state.upper()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return None


@dataclass
class ScanBatchOutcome:
    status: str  # ok | failed | partial
    error_message: str = ""
    failed_count: int = 0
    total_scans: int = 0


def classify_scan_batch(
    samples: List[Sample],
    target_ssid: str,
    target_bssid: Optional[str] = None,
) -> ScanBatchOutcome:
    """Classify a click's sample batch as ok, failed, or partial."""
    if not samples:
        return ScanBatchOutcome(
            status="failed",
            error_message="No scan samples collected",
            failed_count=0,
            total_scans=0,
        )

    by_round: dict[int, List[Sample]] = {}
    for s in samples:
        by_round.setdefault(s.sample_number, []).append(s)

    total = len(by_round)
    failed = 0
    first_error = ""

    for group in by_round.values():
        if all(s.scan_backend == "error" for s in group):
            failed += 1
            if not first_error:
                first_error = next((s.note for s in group if s.note), "Scan failed")
            continue

        valid = [s for s in group if s.rssi_dbm is not None and s.ssid == target_ssid]
        if target_bssid:
            valid = [s for s in valid if s.bssid.upper() == target_bssid.upper()]
        if not valid:
            failed += 1
            if not first_error:
                first_error = next(
                    (s.note for s in group if s.note),
                    "network_not_found",
                )

    if failed == total:
        return ScanBatchOutcome(
            status="failed",
            error_message=first_error or "All scans failed",
            failed_count=failed,
            total_scans=total,
        )
    if failed > 0:
        return ScanBatchOutcome(
            status="partial",
            failed_count=failed,
            total_scans=total,
        )
    return ScanBatchOutcome(status="ok", total_scans=total)


def _signal_to_dbm(signal: int) -> float:
    """Convert nmcli 0-100 signal quality to approximate dBm."""
    return (signal / 2) - 100


def scan_nmcli(
    interface: str,
    ssid: Optional[str] = None,
    bssid: Optional[str] = None,
) -> List[dict]:
    """
    Run nmcli Wi-Fi scan. Returns list of raw AP dicts.
    Fields: ssid, bssid, rssi_dbm, channel, frequency_mhz.
    """
    cmd = [
        "nmcli", "-t",
        "-f", "SSID,BSSID,SIGNAL,CHAN,FREQ,SECURITY",
        "device", "wifi", "list",
        "ifname", interface,
        "--rescan", "yes",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=15)
    except FileNotFoundError:
        raise RuntimeError("nmcli not found")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nmcli error: {e.stderr.strip()}")

    results = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # nmcli -t escapes colons inside field values as \:
        # Split on unescaped colons using a regex
        parts = re.split(r"(?<!\\):", line)
        parts = [p.replace(r"\:", ":") for p in parts]
        if len(parts) < 5:
            continue
        ap_ssid, ap_bssid, signal_str, chan_str, freq_str = parts[:5]
        if ssid and ap_ssid != ssid:
            continue
        if bssid and ap_bssid.upper() != bssid.upper():
            continue
        try:
            sig = int(signal_str)
            rssi = _signal_to_dbm(sig)
        except ValueError:
            rssi = None
        try:
            chan = int(chan_str)
        except ValueError:
            chan = None
        freq = None
        m = re.search(r"(\d+)", freq_str.replace(" ", ""))
        if m:
            freq = float(m.group(1))
        results.append({
            "ssid": ap_ssid,
            "bssid": ap_bssid,
            "rssi_dbm": rssi,
            "channel": chan,
            "frequency_mhz": freq,
        })
    return results


def scan_iw(
    interface: str,
    ssid: Optional[str] = None,
    bssid: Optional[str] = None,
) -> List[dict]:
    """
    Run iw Wi-Fi scan (may need sudo). Returns list of raw AP dicts.
    """
    cmd = ["sudo", "iw", "dev", interface, "scan"]
    last_error = ""
    for attempt in range(1, 4):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=30)
            break
        except FileNotFoundError:
            raise RuntimeError("iw not found")
        except subprocess.CalledProcessError as e:
            last_error = e.stderr.strip()
            if "Device or resource busy" not in last_error or attempt == 3:
                raise RuntimeError(f"iw error: {last_error}")
            time.sleep(float(attempt))

    results = []
    current: dict = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("BSS "):
            if current:
                _maybe_append(current, ssid, bssid, results)
            bss_bssid = line.split()[1].split("(")[0].strip()
            current = {"bssid": bss_bssid, "ssid": "", "rssi_dbm": None,
                       "channel": None, "frequency_mhz": None}
        elif line.startswith("signal:"):
            m = re.search(r"(-?\d+\.?\d*)", line)
            if m:
                current["rssi_dbm"] = float(m.group(1))
        elif line.startswith("freq:"):
            m = re.search(r"(\d+)", line)
            if m:
                current["frequency_mhz"] = float(m.group(1))
        elif line.startswith("* primary channel:") or line.startswith("DS Parameter set: channel"):
            m = re.search(r"(\d+)", line)
            if m:
                current["channel"] = int(m.group(1))
        elif line.startswith("SSID:"):
            current["ssid"] = line[5:].strip()
    if current:
        _maybe_append(current, ssid, bssid, results)
    return results


def _parse_bitrate(line: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s+MBit/s", line, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _parse_mcs(line: str) -> Optional[int]:
    m = re.search(r"(?:VHT-|HE-)?MCS\s+(\d+)", line, re.IGNORECASE)
    return int(m.group(1)) if m else None


def link_iw(interface: str) -> dict:
    """Return negotiated iw link stats when the interface is associated."""
    cmd = ["iw", "dev", interface, "link"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=10)
    except FileNotFoundError:
        raise RuntimeError("iw not found")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"iw link error: {e.stderr.strip()}")

    if "Not connected." in out:
        return {"link_available": False}

    stats = {"link_available": True}
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if line.startswith("Connected to"):
            parts = line.split()
            if len(parts) >= 3:
                stats["connected_bssid"] = parts[2]
        elif line.startswith("signal:"):
            m = re.search(r"(-?\d+)", line)
            if m:
                stats["signal_dbm"] = float(m.group(1))
        elif line.startswith("noise:"):
            m = re.search(r"(-?\d+)", line)
            if m:
                stats["noise_dbm"] = float(m.group(1))
        elif line.lower().startswith("tx bitrate:"):
            stats["tx_bitrate_mbps"] = _parse_bitrate(line)
            stats["tx_mcs"] = _parse_mcs(line)
        elif line.lower().startswith("rx bitrate:"):
            stats["rx_bitrate_mbps"] = _parse_bitrate(line)
            stats["rx_mcs"] = _parse_mcs(line)

    if stats.get("signal_dbm") is not None and stats.get("noise_dbm") is not None:
        stats["snr_db"] = stats["signal_dbm"] - stats["noise_dbm"]
    return stats


def _maybe_append(ap: dict, ssid_filter, bssid_filter, results: list):
    if ssid_filter and ap.get("ssid") != ssid_filter:
        return
    if bssid_filter and ap.get("bssid", "").upper() != bssid_filter.upper():
        return
    results.append(ap)


def scan(
    interface: str,
    ssid: Optional[str] = None,
    bssid: Optional[str] = None,
    backend: str = "iw",
) -> List[dict]:
    """Scan using iw by default, falling back to nmcli only in auto mode."""
    if backend in ("iw", "auto"):
        try:
            return scan_iw(interface, ssid, bssid), "iw"
        except RuntimeError as e:
            if backend == "iw":
                raise
            logger.warning(f"iw failed ({e}); trying nmcli")
    if backend in ("nmcli", "auto"):
        try:
            return scan_nmcli(interface, ssid, bssid), "nmcli"
        except RuntimeError as e:
            if backend == "nmcli":
                raise
            raise RuntimeError(f"iw and nmcli failed; last nmcli error: {e}")
    raise RuntimeError(f"Unknown scan backend: {backend}")


def _is_target_ap(ap: dict, target_ssid: str, target_bssid: Optional[str]) -> bool:
    if ap.get("ssid") != target_ssid:
        return False
    if target_bssid:
        return ap.get("bssid", "").upper() == target_bssid.upper()
    return True


def _mean(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _mode_int(values: list) -> Optional[int]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return max(set(clean), key=clean.count)


def collect_samples(
    interface: str,
    ssid: str,
    bssid: Optional[str] = None,
    samples: int = 10,
    delay_s: float = 0.5,
    click_context: Optional[dict] = None,
    backend: str = "iw",
) -> List[Sample]:
    """
    Run `samples` scans and return all Sample records (one per BSSID per scan).
    click_context: dict with click_id, session_id, router_position_id, x_px, y_px,
                   room_id, room_name, height_ft.
    """
    if click_context is None:
        click_context = {
            "click_id": "cli_test",
            "session_id": "cli_test",
            "router_position_id": "cli_test",
            "x_px": 0,
            "y_px": 0,
            "room_id": "unknown",
            "room_name": "unknown",
            "waypoint_id": "",
            "height_ft": 0,
        }

    all_samples = []
    for i in range(1, samples + 1):
        ts = now_iso()
        link_stats = {}
        try:
            link_stats = link_iw(interface)
        except RuntimeError as e:
            logger.debug(f"iw link unavailable: {e}")

        try:
            # Keep all BSS rows so summary can calculate neighbor interference.
            aps, backend_used = scan(interface, backend=backend)
        except RuntimeError as e:
            logger.error(f"Scan {i} failed: {e}")
            all_samples.append(Sample(
                timestamp=ts,
                session_id=click_context["session_id"],
                router_position_id=click_context["router_position_id"],
                click_id=click_context["click_id"],
                x_px=click_context["x_px"],
                y_px=click_context["y_px"],
                room_id=click_context["room_id"],
                room_name=click_context["room_name"],
                waypoint_id=click_context.get("waypoint_id", ""),
                height_ft=click_context["height_ft"],
                ssid=ssid,
                bssid=bssid or "",
                frequency_mhz=None,
                channel=None,
                rssi_dbm=None,
                interface=interface,
                scan_backend="error",
                sample_number=i,
                link_available=bool(link_stats.get("link_available")),
                noise_dbm=link_stats.get("noise_dbm"),
                snr_db=link_stats.get("snr_db"),
                tx_bitrate_mbps=link_stats.get("tx_bitrate_mbps"),
                rx_bitrate_mbps=link_stats.get("rx_bitrate_mbps"),
                tx_mcs=link_stats.get("tx_mcs"),
                rx_mcs=link_stats.get("rx_mcs"),
                note=str(e),
            ))
            if i < samples:
                time.sleep(delay_s)
            continue

        if not aps:
            all_samples.append(Sample(
                timestamp=ts,
                session_id=click_context["session_id"],
                router_position_id=click_context["router_position_id"],
                click_id=click_context["click_id"],
                x_px=click_context["x_px"],
                y_px=click_context["y_px"],
                room_id=click_context["room_id"],
                room_name=click_context["room_name"],
                waypoint_id=click_context.get("waypoint_id", ""),
                height_ft=click_context["height_ft"],
                ssid=ssid,
                bssid=bssid or "",
                frequency_mhz=None,
                channel=None,
                rssi_dbm=None,
                interface=interface,
                scan_backend=backend_used,
                sample_number=i,
                link_available=bool(link_stats.get("link_available")),
                noise_dbm=link_stats.get("noise_dbm"),
                snr_db=link_stats.get("snr_db"),
                tx_bitrate_mbps=link_stats.get("tx_bitrate_mbps"),
                rx_bitrate_mbps=link_stats.get("rx_bitrate_mbps"),
                tx_mcs=link_stats.get("tx_mcs"),
                rx_mcs=link_stats.get("rx_mcs"),
                note="network_not_found",
            ))
        else:
            for ap in aps:
                is_target = _is_target_ap(ap, ssid, bssid)
                all_samples.append(Sample(
                    timestamp=ts,
                    session_id=click_context["session_id"],
                    router_position_id=click_context["router_position_id"],
                    click_id=click_context["click_id"],
                    x_px=click_context["x_px"],
                    y_px=click_context["y_px"],
                    room_id=click_context["room_id"],
                    room_name=click_context["room_name"],
                    waypoint_id=click_context.get("waypoint_id", ""),
                    height_ft=click_context["height_ft"],
                    ssid=ap.get("ssid", ssid),
                    bssid=ap.get("bssid", ""),
                    frequency_mhz=ap.get("frequency_mhz"),
                    channel=ap.get("channel"),
                    rssi_dbm=ap.get("rssi_dbm"),
                    interface=interface,
                    scan_backend=backend_used,
                    sample_number=i,
                    is_target_ap=is_target,
                    link_available=bool(link_stats.get("link_available")),
                    noise_dbm=link_stats.get("noise_dbm"),
                    snr_db=link_stats.get("snr_db"),
                    tx_bitrate_mbps=link_stats.get("tx_bitrate_mbps"),
                    rx_bitrate_mbps=link_stats.get("rx_bitrate_mbps"),
                    tx_mcs=link_stats.get("tx_mcs"),
                    rx_mcs=link_stats.get("rx_mcs"),
                    note="",
                ))

        if i < samples:
            time.sleep(delay_s)

    return all_samples


def summarize(
    samples: List[Sample],
    target_ssid: str,
    target_bssid: Optional[str] = None,
) -> dict:
    """
    Summarize a list of Sample records for one click point.
    Returns a dict matching SUMMARY_COLUMNS.
    """
    import math
    import statistics

    filtered = [s for s in samples if s.is_target_ap or s.ssid == target_ssid]
    if target_bssid:
        filtered = [s for s in filtered if s.bssid.upper() == target_bssid.upper()]

    valid = [s for s in filtered if s.rssi_dbm is not None]
    by_round: dict[int, List[Sample]] = {}
    for sample in samples:
        by_round.setdefault(sample.sample_number, []).append(sample)
    missing = sum(
        1
        for group in by_round.values()
        if not any(s.rssi_dbm is not None and (s.is_target_ap or s.ssid == target_ssid) for s in group)
    )

    if not valid:
        rssi_avg = rssi_min = rssi_max = rssi_std = None
        best_bssid = target_bssid or ""
        freq = channel = None
    else:
        # Group by bssid to find strongest average
        bssid_groups: dict[str, list] = {}
        for s in valid:
            bssid_groups.setdefault(s.bssid, []).append(s.rssi_dbm)
        best_bssid = max(bssid_groups, key=lambda b: sum(bssid_groups[b]) / len(bssid_groups[b]))

        rssi_values = [s.rssi_dbm for s in valid]
        rssi_avg = sum(rssi_values) / len(rssi_values)
        rssi_min = min(rssi_values)
        rssi_max = max(rssi_values)
        rssi_std = statistics.stdev(rssi_values) if len(rssi_values) > 1 else 0.0

        # Use metadata from the last valid sample
        last = valid[-1]
        freq = last.frequency_mhz
        channel = last.channel

    round_link_rows = []
    for group in by_round.values():
        link_row = next((s for s in group if s.link_available), None)
        if link_row:
            round_link_rows.append(link_row)

    noise_avg = _mean([s.noise_dbm for s in round_link_rows])
    snr_avg = _mean([s.snr_db for s in round_link_rows])
    snr_values = [s.snr_db for s in round_link_rows if s.snr_db is not None]
    snr_min = min(snr_values) if snr_values else None
    tx_avg = _mean([s.tx_bitrate_mbps for s in round_link_rows])
    rx_avg = _mean([s.rx_bitrate_mbps for s in round_link_rows])
    tx_mcs_mode = _mode_int([s.tx_mcs for s in round_link_rows])
    rx_mcs_mode = _mode_int([s.rx_mcs for s in round_link_rows])

    target_freq = freq
    neighbors = [
        s
        for s in samples
        if not (s.is_target_ap or (target_bssid and s.bssid.upper() == target_bssid.upper()))
        and s.rssi_dbm is not None
        and s.rssi_dbm >= -80
    ]
    same_channel_by_bssid = {
        s.bssid: s
        for s in neighbors
        if channel is not None and s.channel == channel
    }
    same_channel = list(same_channel_by_bssid.values())
    adjacent_by_bssid = {}
    for s in neighbors:
        if channel is None or s.channel is None or s.channel == channel:
            continue
        if target_freq and target_freq < 3000:
            if abs(s.channel - channel) <= 1:
                adjacent_by_bssid[s.bssid] = s
        elif target_freq and s.frequency_mhz:
            if abs(s.frequency_mhz - target_freq) <= 40:
                adjacent_by_bssid[s.bssid] = s
    adjacent = list(adjacent_by_bssid.values())

    if same_channel:
        mw_sum = sum(10 ** (s.rssi_dbm / 10) for s in same_channel)
        neighbor_rssi_sum = 10 * math.log10(mw_sum) if mw_sum > 0 else None
    else:
        neighbor_rssi_sum = None
    channel_utilization_proxy = min(100, len(same_channel) * 5 + len(adjacent) * 2)

    ctx = samples[0] if samples else None
    return {
        "timestamp_start": samples[0].timestamp if samples else now_iso(),
        "timestamp_end": samples[-1].timestamp if samples else now_iso(),
        "session_id": ctx.session_id if ctx else "",
        "router_position_id": ctx.router_position_id if ctx else "",
        "click_id": ctx.click_id if ctx else "",
        "x_px": ctx.x_px if ctx else 0,
        "y_px": ctx.y_px if ctx else 0,
        "room_id": ctx.room_id if ctx else "",
        "room_name": ctx.room_name if ctx else "",
        "waypoint_id": ctx.waypoint_id if ctx else "",
        "height_ft": ctx.height_ft if ctx else 0,
        "target_ssid": target_ssid,
        "target_bssid": target_bssid or "",
        "best_bssid": best_bssid,
        "frequency_mhz": freq,
        "channel": channel,
        "sample_count": len(valid),
        "rssi_avg_dbm": round(rssi_avg, 2) if rssi_avg is not None else None,
        "rssi_min_dbm": round(rssi_min, 2) if rssi_min is not None else None,
        "rssi_max_dbm": round(rssi_max, 2) if rssi_max is not None else None,
        "rssi_std_db": round(rssi_std, 3) if rssi_std is not None else None,
        "noise_avg_dbm": round(noise_avg, 2) if noise_avg is not None else None,
        "snr_avg_db": round(snr_avg, 2) if snr_avg is not None else None,
        "snr_min_db": round(snr_min, 2) if snr_min is not None else None,
        "tx_bitrate_avg_mbps": round(tx_avg, 2) if tx_avg is not None else None,
        "rx_bitrate_avg_mbps": round(rx_avg, 2) if rx_avg is not None else None,
        "tx_mcs_mode": tx_mcs_mode,
        "rx_mcs_mode": rx_mcs_mode,
        "neighbor_count_same_channel": len(same_channel),
        "neighbor_count_adjacent": len(adjacent),
        "neighbor_rssi_sum_dbm": round(neighbor_rssi_sum, 2) if neighbor_rssi_sum is not None else None,
        "channel_utilization_proxy": channel_utilization_proxy,
        "link_available": bool(round_link_rows),
        "missing_sample_count": missing,
        "note": "",
    }


def _pretty_print(samples: List[Sample], summary: dict):
    print("\n=== Raw Samples ===")
    print(f"{'#':<4} {'BSSID':<20} {'SSID':<25} {'RSSI':>8}  {'Chan':>4}  {'Freq':>8}")
    print("-" * 75)
    for s in samples:
        rssi = f"{s.rssi_dbm:.1f}" if s.rssi_dbm is not None else "N/A"
        chan = str(s.channel) if s.channel else "?"
        freq = f"{s.frequency_mhz:.0f}" if s.frequency_mhz else "?"
        print(f"{s.sample_number:<4} {s.bssid:<20} {s.ssid:<25} {rssi:>8}  {chan:>4}  {freq:>8}")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Wi-Fi scan CLI")
    parser.add_argument("--interface", required=True, help="Wi-Fi interface (e.g. wlan1)")
    parser.add_argument("--ssid", required=True, help="Target SSID")
    parser.add_argument("--bssid", default=None, help="Optional target BSSID filter")
    parser.add_argument("--samples", type=int, default=10, help="Number of scans")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between scans (s)")
    parser.add_argument("--backend", choices=["auto", "nmcli", "iw"], default="iw")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ifaces = list_wifi_interfaces()
    print(f"Detected Wi-Fi interfaces: {ifaces}")

    print(f"\nCollecting {args.samples} samples from {args.interface} for SSID '{args.ssid}'...")
    samples = collect_samples(
        interface=args.interface,
        ssid=args.ssid,
        bssid=args.bssid,
        samples=args.samples,
        delay_s=args.delay,
        backend=args.backend,
    )
    summary = summarize(samples, args.ssid, args.bssid)
    _pretty_print(samples, summary)


if __name__ == "__main__":
    main()
