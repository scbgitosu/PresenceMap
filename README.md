# PresenceMap

ESP32 CSI room-presence sensing on your Mac. Three RuView-flashed **ESP32-S3** nodes stream WiFi channel state information over UDP; PresenceMap ingests, diagnoses link health, records labeled sessions, and trains a small sequence model for occupied / vacant inference.

**North star:** privacy-first **sleep tracking** in the bedroom—movement, restlessness, respiration and heart rate (where CSI supports it), and an overall **sleep score**—validated against trusted wearables (e.g. Apple Watch), not marketed as clinical vitals.

## Sleep tracking roadmap

PresenceMap is intentionally staged. Each phase produces something you can measure before adding complexity.

| Phase | Goal | Success criteria | Status |
|-------|------|------------------|--------|
| **0 — Link health** | Reliable raw CSI ingest, 3-node bed cluster | ESP32 Nodes: all **ok**, CSI frames climbing, PPS stable | **You are here** (ingest working) |
| **1 — Bed presence & motion** | Know *in bed* vs *out of bed*; coarse movement at night | Train `TemporalCSIModel` on labeled vacant/occupied sessions; live inference stable overnight | **Training pipeline works** (see below); labels are manual phase tags today |
| **2 — Night timeline** | Segment the night: in bed, awake, restless, still | Align CSI windows to time; optional **Apple Watch** sleep stages as reference labels (import TBD) | Not built |
| **3 — Vitals from CSI** | Respiration & heart rate tracks vs watch | Compare to Watch HR / respiratory rate during sleep; report confidence bands, not “medical grade” | Not built (RuView edge vitals are not ingested by PresenceMap v3) |
| **4 — Sleep score** | One nightly score + explainable factors | Calibrated on watch sleep score + your notes; honest limits documented | Not built |

### Milestones along the way

1. **Record a week of nights** — `labeled_vacant` (empty room), `labeled_occupied` (you in bed, still), `labeled_occupied_moving` (reading / shifting). Same Wi‑Fi layout each night.
2. **First bed model** — Train on 2+ session types; run **ESP32 live inference** overnight; log false wakes vs how it felt.
3. **Watch baseline export** — Export sleep from Apple Health (XML or third-party CSV). Build a small importer that aligns timestamps to CSI `rf_windows.parquet` (planned; not in repo yet).
4. **Restlessness metric** — Use the model’s **motion head** and amplitude deltas as a restless index; plot vs watch “awake” minutes.
5. **Respiration / HR experiments** — CSI phase stability in breathing band (0.1–0.5 Hz) and faster components; cross-check watch HR only where literature and your data agree.
6. **Sleep score v0** — Weighted blend (time in bed, movement, optional vitals agreement with watch); never claim clinical accuracy.

### Apple Watch as baseline

Using your Watch is a strong idea for **labels and validation**, not as the runtime sensor:

- **Good for:** sleep interval boundaries, awake vs asleep, HR and respiratory rate trends, a target sleep score to correlate against.
- **Limits:** Watch is on your wrist, CSI sees the **bed volume**; arm movement ≠ torso breathing; export lag and Apple’s stage model are proprietary.
- **Product rule:** Present CSI-derived vitals as **estimates with confidence**, same as [AGENTS.md](AGENTS.md)—do not claim clinical-grade HR/SpO₂/sleep staging out of the box.

### What CSI can and cannot do (honest)

| Signal | CSI (3 nodes, bed) | Watch |
|--------|-------------------|--------|
| In bed / out of bed | Strong candidate (Phase 1) | Indirect |
| Large movement / restlessness | Strong (motion head + amp deltas) | Accelerometer |
| Respiration | Possible, needs validation | Often available in sleep |
| Heart rate | Harder through CSI; research-grade caution | Primary on wrist |
| Sleep stages / score | Derived, model + watch calibration | Reference label |

## Is training functional?

**Yes, for Phase 1 (binary occupancy + motion auxiliary loss)—not for full sleep score or vitals yet.**

| Capability | Works today? |
|------------|----------------|
| Record sessions → `sessions/<id>/rf_windows.parquet` | Yes (`esp32-record`, dashboard **Record**) |
| Phase labels → `labels.parquet` | Yes (`labeled_vacant`, `labeled_occupied`, `labeled_occupied_moving`, `calibration`) |
| **Training → Train** in dashboard | Yes — builds dataset, trains on MPS, saves model + metrics |
| CLI `presence-mac train` | Yes |
| **Models** browser, eval summary | Yes |
| **ESP32 live inference** | Yes (needs ingest + trained model) |
| Apple Watch import / alignment | No |
| Sleep stages, respiration, HR, sleep score models | No |

Minimum practical train: **two session types** (e.g. vacant + occupied), enough windows (dozens+ per class). The trainer errors if labeled windows &lt; 8.

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

Keep ingest running while recording, or let `esp32-record` bind UDP alone for that session.

```sh
presence-mac esp32-record --project data/survey_projects/my_bed \
  --session bed_vacant_001 --duration 120 --phase labeled_vacant

presence-mac esp32-record --project data/survey_projects/my_bed \
  --session bed_occ_001 --duration 120 --phase labeled_occupied

presence-mac esp32-record --project data/survey_projects/my_bed \
  --session bed_restless_001 --duration 120 --phase labeled_occupied_moving
```

Then in the dashboard: **Training → Train** (select sessions, **Build dataset + train**), **Models** to review metrics, **ESP32 Nodes → ESP32 live inference** for overnight trials.

CLI equivalent:

```sh
presence-mac train --project data/survey_projects/my_bed \
  --sessions bed_vacant_001 bed_occ_001 --epochs 30
```

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
| Apple Watch (optional) | Reference labels for sleep roadmap phases 2–4 |

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

## See also

- [docs/ROADMAP.md](docs/ROADMAP.md) — v3 platform features and retired paths
