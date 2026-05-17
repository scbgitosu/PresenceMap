# PresenceMap Roadmap

## Milestone 0: Local Fork Bootstrap

- Preserve HeatMap history in a separate local repo.
- Rename the project conceptually to PresenceMap.
- Keep the HeatMap remote as an upstream reference only.

## Milestone 1: Tripwire Prototype

Build a headless HP collector that samples continuously and writes:

- `presence_raw.csv`: every observation window
- `presence_events.csv`: detected motion or presence state changes

Initial detection features:

- RSSI rolling mean and standard deviation
- RSSI delta from baseline
- SNR delta from baseline
- link bitrate and MCS changes when available
- visible BSSID count and channel changes

Initial states:

- `unknown`
- `vacant`
- `motion`
- `occupied`

## Milestone 2: Calibration Workflow

Add a guided workflow for collecting labeled examples:

- vacant apartment baseline
- doorway crossing
- hallway walk
- single-room occupied
- multi-room occupied

The first implementation should prefer transparent thresholds and charts over a
black-box model.

## Milestone 3: Mac Dashboard

Adapt the existing Streamlit dashboard patterns to show:

- live or recently synced event timeline
- floorplan room state
- per-BSSID signal traces
- baseline comparison
- false positive / false negative review notes

## Milestone 4: Automation Output

Add optional local integrations:

- MQTT publish
- Home Assistant webhook
- local JSONL event stream
- simple shell command hook

## Open Technical Questions

- How stable are the Nighthawk router and node RSSI observations over long
  vacant periods?
- Does the Atheros adapter expose reliable enough SNR/noise/link metrics?
- Is monitor mode useful on this adapter without destabilizing ordinary scans?
- Can two central RF paths produce room-level occupancy estimates, or is the
  signal only reliable for doorway/hallway tripwires?
- What is the minimum sampling interval that still catches motion without
  making the Wi-Fi stack noisy?
