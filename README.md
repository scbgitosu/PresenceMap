# PresenceMap

PresenceMap is a local-first RF sensing experiment for exploring whether ordinary
Wi-Fi observations can support motion detection, room presence, and lightweight
automation/security workflows.

This project is forked from HeatMap and intentionally keeps the same core stack:

- HP Linux machine for field collection
- External Atheros Wi-Fi adapter
- Python collector and analysis tools
- CSV/event-log pipeline
- Mac-side Streamlit dashboards
- Floorplan, room, and session metadata

The goal is different from HeatMap. HeatMap maps Wi-Fi quality to choose access
point placement. PresenceMap watches how Wi-Fi observations change over time and
tries to infer events such as movement through a doorway, room occupancy, and
vacancy.

## Hardware Target

Initial setup:

- Netgear Nighthawk router
- Netgear Nighthawk node
- HP Linux computer
- External Atheros network interface
- MacBook M1 Max Pro for analysis and dashboard work

## First Hypothesis

The first realistic milestone is a Wi-Fi tripwire:

1. Establish a stable baseline across a doorway, hallway, or room boundary.
2. Continuously sample RSSI, SNR, bitrate, MCS, channel, and visible BSSIDs.
3. Detect short-window changes that look like a body crossing the RF path.
4. Log motion events with confidence and timestamps.
5. Review those events on a Mac dashboard.

Whole-apartment room occupancy is a second milestone. It may be possible to infer
coarse presence with careful calibration, but it should be treated as a
confidence-scored estimate rather than precise tracking.

## Project Direction

PresenceMap will evolve the original HeatMap workflow in three phases:

### Phase 1: Continuous Collection

- Add an HP-side continuous collector.
- Record time-series RF observations instead of click-based survey points.
- Keep the existing interface discovery, `iw`/`nmcli` support, project config,
  floorplan, room metadata, and CSV writer patterns.

### Phase 2: Motion and Presence Scoring

- Build baseline profiles for vacant, occupied, and movement states.
- Compute rolling-window deltas and variance.
- Emit event rows such as `motion`, `occupied`, `vacant`, and `unknown`.
- Keep results explainable before trying any heavier modeling.

### Phase 3: Automation Hooks

- Publish events to MQTT, Home Assistant webhooks, or a local API.
- Support rules such as turning on lights when confidence crosses a threshold.
- Keep an auditable event log for security-context experiments.

## Current Status

This repo now contains the inherited HeatMap survey workflow plus a first
PresenceMap prototype for headless HP-side tripwire collection.

## HP Presence Tripwire Prototype

The prototype collector runs on the HP Linux machine with the external Atheros
adapter. It reuses `hp_collector/wifi_scan.py` for `iw`/`nmcli` scanning, then
writes a continuous raw log and a motion-event log under the selected project:

```text
survey_projects/<project>/presence_sessions/<session>/
  presence_raw.csv
  presence_events.csv
  presence_baseline.json
```

`presence_raw.csv` contains every observed BSSID row from each scan window,
including RSSI, channel, link RSSI/SNR/bitrate/MCS fields when available, plus
window-level tripwire metadata. `presence_events.csv` contains thresholded
`motion` events with confidence scores and baseline deltas. This is coarse RF
motion sensing only; it does not identify people.

### HP Linux Setup

On the HP, install system tools and Python dependencies from the repo root:

```bash
sudo apt update
sudo apt install network-manager iw python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-hp.txt
```

Plug in the Atheros adapter and find its interface name:

```bash
iw dev
nmcli device status
```

The examples below use `wlan1`; replace it with the detected Atheros interface.
For the default `iw` backend, the scanner invokes `sudo iw dev <iface> scan`, so
run from a terminal where `sudo` is available.

### Calibrate a Baseline

Place the HP and adapter in the intended tripwire position, keep the doorway or
room boundary vacant, then collect a baseline:

```bash
./scripts/run_presence_tripwire.sh \
  --project survey_projects/apartment_test \
  --session front_door_tripwire \
  --interface wlan1 \
  --calibrate \
  --baseline-seconds 120 \
  --location-label front_door
```

This writes `presence_baseline.json` and also appends calibration observations
to `presence_raw.csv`. Recalibrate whenever the adapter, router/node placement,
target SSID/BSSID, or tripwire location changes.

### Run Continuous Monitoring

After calibration, run the headless monitor:

```bash
./scripts/run_presence_tripwire.sh \
  --project survey_projects/apartment_test \
  --session front_door_tripwire \
  --interface wlan1 \
  --monitor
```

Useful tuning flags:

```bash
--samples-per-window 5      # scans per scoring window
--window-seconds 5          # approximate cadence
--threshold 2.5             # higher is less sensitive
--cooldown-seconds 10       # minimum spacing between event rows
--backend auto              # try iw, then nmcli fallback
--bssid aa:bb:cc:dd:ee:ff  # lock tripwire to one AP/router/node
```

For a short smoke test without leaving it running:

```bash
./scripts/run_presence_tripwire.sh \
  --project survey_projects/apartment_test \
  --session front_door_tripwire \
  --interface wlan1 \
  --monitor \
  --max-windows 3
```

### Train Whole-Home Occupancy

Whole-home occupancy is trained from guided labeled blocks. Start with a vacant
home block, then collect occupied-still and occupied-moving blocks in the same
presence session:

```bash
./scripts/run_presence_tripwire.sh \
  --project survey_projects/apartment_test \
  --session home_occupancy \
  --interface wlan1 \
  --label-block vacant \
  --block-seconds 300

./scripts/run_presence_tripwire.sh \
  --project survey_projects/apartment_test \
  --session home_occupancy \
  --interface wlan1 \
  --label-block occupied_still \
  --block-seconds 300

./scripts/run_presence_tripwire.sh \
  --project survey_projects/apartment_test \
  --session home_occupancy \
  --interface wlan1 \
  --label-block occupied_moving \
  --block-seconds 300
```

This writes feature, label, and model artifacts beside the tripwire logs:

```text
survey_projects/<project>/presence_sessions/<session>/
  presence_features.csv
  presence_labels.csv
  presence_states.csv
  presence_model.json
```

Run conservative live occupancy scoring after the model has both vacant and
occupied examples:

```bash
./scripts/run_presence_tripwire.sh \
  --project survey_projects/apartment_test \
  --session home_occupancy \
  --interface wlan1 \
  --monitor \
  --occupancy-monitor
```

The live state stream is intentionally conservative. It can emit `unknown` when
the RF evidence is weak or mixed, and it does not identify people.

## Safety and Privacy

PresenceMap should be treated as experimental sensing infrastructure. It should
not be used as a sole security system, and any occupancy logging should be
designed with clear local control, retention limits, and visibility into what is
being stored.
