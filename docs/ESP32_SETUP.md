# ESP32 CSI setup (Mac + RuView firmware)

PresenceMap ingests **ADR-018** binary CSI frames from RuView-flashed ESP32-S3 nodes over UDP port **5005**. This guide covers a **three-node bedroom cluster** on a Mac (Apple Silicon) with a Nighthawk router.

## Prerequisites

| Item | Notes |
|------|--------|
| 3× ESP32-S3 (8 MB flash recommended) | Original ESP32 / ESP32-C3 are **not** supported by RuView firmware |
| RuView firmware | Flash from [PresenceMap/RuView](../RuView/) per RuView `docs/user-guide.md` |
| Mac on same Wi‑Fi as nodes | Provision `--target-ip` = Mac **LAN** IP (not Tailscale unless you route UDP) |
| Python env | `pip install -e .` from repo root |

## Provision nodes

From the RuView tree (example):

```bash
cd PresenceMap/RuView
python firmware/esp32-csi-node/provision.py --port /dev/cu.usbserial-XXXX \
  --ssid "YourSSID" --password "secret" --target-ip 192.168.1.42
```

Repeat for each node. Assign distinct node IDs if your provision script supports it.

Find your Mac LAN IP:

```bash
ipconfig getifaddr en0
```

## Before starting PresenceMap

1. **Stop other UDP consumers** on port 5005 (RuView `sensing-server`, Docker `wifi-densepose`, hardware `aggregator`). Only one process may bind `:5005`.
2. **Power-cycle** each node after first flash if Wi‑Fi connects but no CSI (see [RuView TROUBLESHOOTING §1](../RuView/docs/TROUBLESHOOTING.md)).
3. **Verify wire traffic**:

```bash
sudo tcpdump -i en0 -n udp port 5005
```

Expect **three source IPs**, steady datagrams. RuView firmware may multiplex **sibling** packet types (vitals, feature state) on the same port; PresenceMap skips those and parses raw CSI (`magic 0xC5110001`).

4. **Do not run RuView in Docker on Windows** for multi-node UDP (known limitation). On macOS, run PresenceMap **natively**.

## Run diagnostics (Phase 0)

Terminal 1 — ingest + runtime state file:

```bash
presence-mac esp32-ingest \
  --project data/survey_projects/my_bed \
  --udp-port 5005
```

Terminal 2 — dashboard:

```bash
presence-mac dashboard --project data/survey_projects/my_bed
```

Open **Live → ESP32 Nodes**. All three nodes should show **ok** (last packet &lt; 2 s). Aggregate packet rate should be stable without freezing the UI.

Runtime state is written to `data/runtime/esp32_state.json` (gitignored).

## Record a short session (for later training)

```bash
presence-mac esp32-record \
  --project data/survey_projects/my_bed \
  --session esp32_diag_001 \
  --duration 60 \
  --phase calibration
```

Or use **Record 60s** on the ESP32 Nodes dashboard page (requires ingest running in another terminal).

## Health thresholds

| Status | Condition |
|--------|-----------|
| ok | Last frame &lt; 2 s ago |
| stale | 2–10 s |
| offline | &gt; 10 s |

Sequence-gap ratio &gt; 5% per node usually means Wi‑Fi congestion, wrong target IP, or a second consumer stealing packets.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| tcpdump sees packets, UI offline | Wrong `target-ip`; Mac firewall blocking UDP 5005; ingest not running |
| One node only | Second process on 5005; nodes share same IP (NAT); re-provision |
| High seq gaps | Reduce nodes streaming raw CSI + feature packets; move nodes closer to AP |
| Parser errors in ingest log | Firmware sending only feature-state packets; reflash RuView CSI firmware or enable raw CSI stream |

## Reference

- Wire format: [RuView ADR-018](../RuView/docs/adr/ADR-018-esp32-dev-implementation.md)
- Upstream ops: [RuView TROUBLESHOOTING](../RuView/docs/TROUBLESHOOTING.md)
