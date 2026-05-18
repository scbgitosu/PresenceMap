#!/usr/bin/env bash
# Atheros CSI Tool bring-up for the HP sensor agent.
#
# Assumes Ubuntu 22.04 LTS with the patched ath9k driver already installed
# (see docs/AGENT_SETUP_LINUX.md for the kernel + firmware steps). This script
# does the per-boot bits: unblock wifi, reload ath9k, put the AR9271 in
# monitor mode on the configured channel, and report status.
#
# Usage:
#   sudo ./tools/scripts/atheros_csi_setup.sh [--iface wlan1] [--channel 6] [--bw HT20]

set -euo pipefail

IFACE="wlan1"
CHANNEL="6"
BW="HT20"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iface)   IFACE="$2"; shift 2 ;;
    --channel) CHANNEL="$2"; shift 2 ;;
    --bw)      BW="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "must be run as root (sudo)" >&2
  exit 1
fi

echo "[csi-setup] unblock wifi"
rfkill unblock wifi || true

echo "[csi-setup] reload ath9k_htc"
modprobe -r ath9k_htc ath9k 2>/dev/null || true
sleep 0.5
modprobe ath9k_htc

# Wait for the interface to appear (USB enumerate can take a moment).
for i in 1 2 3 4 5; do
  if ip link show "$IFACE" >/dev/null 2>&1; then
    break
  fi
  echo "[csi-setup] waiting for $IFACE..."
  sleep 1
done

if ! ip link show "$IFACE" >/dev/null 2>&1; then
  echo "[csi-setup] $IFACE never appeared. Plug in the AR9271 USB adapter and try again." >&2
  exit 3
fi

echo "[csi-setup] $IFACE -> monitor mode on channel $CHANNEL ($BW)"
ip link set "$IFACE" down
iw dev "$IFACE" set type monitor
ip link set "$IFACE" up
iw dev "$IFACE" set channel "$CHANNEL" "$BW"

echo "[csi-setup] interface state:"
iw dev "$IFACE" info | sed -n 's/^/  /p'

echo "[csi-setup] done. To start capturing CSI:"
echo "  presence-agent csi-test --project data/survey_projects/apartment_test --iface $IFACE --duration 5 --source atheros"
