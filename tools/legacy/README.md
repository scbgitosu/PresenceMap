# tools/legacy/

Reference-only HeatMap survey code and pre-v2 PresenceMap modules. **Nothing
under ``mac_app/`` or ``hp_agent/`` imports from here.** Files were moved
verbatim from ``mac_analysis/`` and ``shared/`` so their git history is preserved.

## What's here

- ``heatmap_generator.py`` — RSSI heatmap renderer (was ``mac_analysis/``).
- ``placement_optimizer.py`` — path-loss-based AP placement suggestions.
- ``session_compare.py`` — multi-session metrics ranking.
- ``walk_matcher.py`` — aligns measurement clicks across sessions.
- ``survey_metrics.py`` — point/grid/room metric computations.
- ``path_loss.py`` — least-squares path-loss model fit.
- ``session_cleanup.py`` — survey-session artifact cleanup utility.
- ``occupancy_v1.py`` — v1 Gaussian-profile occupancy training/scoring
  (was ``shared/occupancy.py``).
- ``webcam_motion_v1.py`` — v1 motion-ratio webcam ground truth
  (was ``shared/webcam_ground_truth.py``).

## Heads-up: stale imports

Some files here still ``from shared.utils import project_paths`` (moved to
``shared.project``) or ``from shared.csv_schema import ...`` (deleted). They
will not run as-is. Update imports if you revive any of them. Nothing in the
live tree references this code, so the build is unaffected.
