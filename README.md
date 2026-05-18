# PresenceMap v2

RF-based room-presence detection. A thin Linux **sensor agent** captures CSI
(Channel State Information) from an Atheros AR9271 in monitor mode plus
RSSI/SNR from `iw scan`, and streams it to a **Mac dashboard** that owns
training (with webcam-derived YOLO ground truth), live inference (on MPS),
and a local REST API.

## Architecture

```
   HP laptop (Ubuntu)                            Mac (Apple Silicon)
   ┌──────────────────────────┐                ┌─────────────────────────────┐
   │   presence-agent run     │  ZMQ pub/sub   │   presence-mac dashboard    │
   │ ─────────────────────── │ ─────────────► │ Streamlit (Training + Live) │
   │   AR9271 monitor mode    │   tcp/5555     │     SessionRecorder         │
   │   recvCSI subprocess     │   tcp/5556     │     YOLOv8n on MPS          │
   │   iw scan -> RSSI/SNR    │                │     InferenceLoop on MPS    │
   │   webcam JPEG @ 5 Hz     │                │     LiveBuffer (SQLite)     │
   │   per-window WindowMsg   │                │   presence-mac api          │
   │   + HealthMsg every 2 s  │                │   FastAPI /state /history   │
   └──────────────────────────┘                └─────────────────────────────┘
                                                   ▲
                                                   │ same SQLite
                                                   ▼
                                              data/runtime/live_buffer.sqlite3
```

The HP never runs the model. The Mac never runs the radio. Everything is in
the open in `data/survey_projects/<project>/` and `data/models/`.

## Quickstart

### HP (Linux sensor)

One-time setup is in [docs/AGENT_SETUP_LINUX.md](docs/AGENT_SETUP_LINUX.md)
(patched ath9k driver, `recvCSI` binary, passwordless sudo for `iw`).

Per-boot:

```sh
sudo ./tools/scripts/atheros_csi_setup.sh --iface wlan1 --channel 6 --bw HT20
```

Run the agent:

```sh
presence-agent run \
  --project ../data/survey_projects/apartment_test \
  --session train001 \
  --csi --csi-source atheros
```

### Mac (Training + Live)

```sh
pip install -e .
pip install -r requirements-mac.txt
```

Training:

```sh
presence-mac dashboard --project data/survey_projects/apartment_test
```

In the browser:

1. **Training → Setup**: ping the HP agent.
2. **Training → Collect**: start a session, switch phases (calibration →
   labeled_vacant → labeled_occupied), stop. YOLOv8n on MPS labels frames at
   2 Hz; per-window labels and rolling thumbnails land under
   `data/survey_projects/<project>/sessions/<id>/`.
3. **Training → Train**: pick session(s), click Train. ~60k-param GRU on MPS
   typically finishes in seconds; model + eval are saved under
   `data/models/<model_id>/`.

Or via CLI:

```sh
presence-mac train --project data/survey_projects/apartment_test --session train001
```

Live mode:

```sh
# In one terminal -- the local REST API
presence-mac api --project data/survey_projects/apartment_test

# In another -- the dashboard (Live tab)
presence-mac dashboard --project data/survey_projects/apartment_test
```

REST endpoints:

```sh
curl http://127.0.0.1:8765/state
curl 'http://127.0.0.1:8765/history?since=2026-05-18T00:00:00Z&limit=200'
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/models
```

## Why CSI + YOLO?

The v1 pipeline trained on RSSI alone with motion-ratio webcam labels.
[.claude/plans/check-the-current-implementation-sprightly-pixel.md](.claude/plans/check-the-current-implementation-sprightly-pixel.md)
documents what went wrong; the short version:

- v1's `link_iw()` silently dropped every `iw link` failure, so 100% of
  SNR/noise values were NULL. Models were RSSI-only.
- The motion-ratio webcam labels couldn't tell "still occupant" from
  "empty room", so labels were 75% vacant / 3% occupied_moving on real
  sessions. v2 swaps in YOLO person detection.
- CSI gives per-subcarrier amplitude and phase, which captures multipath
  fading directly. A small sequence model on top is the cheapest big
  accuracy unlock you get from minimal hardware.

## Layout

```
hp_agent/       Linux sensor: CSI + RSSI + webcam + ZMQ publishers
mac_app/        Mac side: dashboard, training, live inference, REST API
shared/         Shared dataclasses, schemas, project paths
tools/legacy/   Archived v1 / HeatMap survey code -- not imported anywhere
data/           Projects, sessions, models, runtime SQLite (gitignored)
docs/           Architecture + Linux setup + roadmap
tests/          pytest suite
```

## License

See repo root.
