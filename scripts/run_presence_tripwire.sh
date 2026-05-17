#!/usr/bin/env bash
# Headless launcher for the PresenceMap HP-side Wi-Fi tripwire collector.
#
# Usage:
#   ./scripts/run_presence_tripwire.sh --project survey_projects/apartment_test --calibrate
#   ./scripts/run_presence_tripwire.sh --project survey_projects/apartment_test --monitor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log()  { printf '[presence_tripwire] %s\n' "$*" >&2; }
fail() { printf '[presence_tripwire] ERROR: %s\n' "$*" >&2; exit 1; }

if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
    log "activated .venv"
fi

if ! command -v python >/dev/null 2>&1; then
    fail "python not found on PATH. Create the venv per README and re-run."
fi

if ! command -v iw >/dev/null 2>&1; then
    fail "iw not found. Install HP packages with: sudo apt install network-manager iw"
fi

has_project=0
for arg in "$@"; do
    if [[ "${arg}" == "--project" || "${arg}" == --project=* ]]; then
        has_project=1
        break
    fi
done
if (( has_project == 0 )); then
    set -- --project survey_projects/apartment_test "$@"
    log "no --project supplied; defaulting to survey_projects/apartment_test"
fi

python hp_collector/presence_tripwire.py "$@"
