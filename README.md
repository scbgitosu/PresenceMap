# PresenceMap

ESP32 CSI room-presence sensing on your Mac. Three RuView-flashed **ESP32-S3** nodes stream WiFi channel state information over UDP; PresenceMap ingests, diagnoses link health, records labeled sessions, and trains a small sequence model for occupied / vacant inference.

## Quickstart

```sh
pip install -e .
pip install -r requirements-mac.txt
```

**Terminal 1** — ingest (must be the only process on UDP 5005):

```sh
presence-mac esp32-ingest --project data/survey_projects/my_bed
```

**Terminal 2** — dashboard:

```sh
presence-mac dashboard --project data/survey_projects/my_bed
```

Open **ESP32 Nodes** and confirm all nodes show **ok**.

## Record + train

Stop ingest, then record labeled sessions:

```sh
presence-mac esp32-record --project data/survey_projects/my_bed \
  --session bed_vacant_001 --duration 120 --phase labeled_vacant

presence-mac esp32-record --project data/survey_projects/my_bed \
  --session bed_occ_001 --duration 120 --phase labeled_occupied
```

Train and run live inference from the dashboard (**Train**, **Models**, **ESP32 Nodes**).

Optional REST API:

```sh
presence-mac api --project data/survey_projects/my_bed
```

## Hardware

| Item | Role |
|------|------|
| 3× ESP32-S3 | CSI nodes (RuView firmware, UDP → Mac) |
| Mac (Apple Silicon) | Ingest, UI, training on MPS |
| Wi‑Fi router | AP for nodes + home network |

See [docs/ESP32_SETUP.md](docs/ESP32_SETUP.md). Wire format and firmware live in [RuView/](RuView/) (upstream reference, not modified by PresenceMap).

## Layout

```
mac_app/
  capture/     ESP32 UDP ingest, ADR-018 parser, CSI features
  train/       TemporalCSIModel, dataset, registry
  inference/   Live buffer, ESP32 inference loop
  dashboard/   Streamlit UI
  api/         Local REST service
shared/        Project paths, schemas
data/          Projects, sessions, models (gitignored artifacts)
RuView/        Upstream CSI firmware + docs (reference)
docs/
tests/
```

## What we removed

PresenceMap no longer includes the HeatMap survey stack, HP Linux `presence-agent`, AR9271 / Atheros CSI tool path, ZMQ HP→Mac transport, YOLO webcam labeling, or legacy floorplan survey UI. Those lived in earlier v1/v2 prototypes.
