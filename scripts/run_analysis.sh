#!/usr/bin/env bash
#
# run_analysis.sh — orchestrate the UltraChat 200k analysis pipeline end-to-end.
#
# Runs each analysis stage in order (as `python -m pipeline.analysis.<stage>`),
# timing each, and writes the final academic report to
# $UC_RESULTS_DIR/ANALYSIS_REPORT.md. Any stage failure aborts the run.
#
# Usage (run from anywhere; the script cd's to the repo root):
#   bash scripts/run_analysis.sh                  # install deps, download, analyse, report
#   bash scripts/run_analysis.sh --skip-download  # reuse parquet shards already in UC_DATA_DIR
#   bash scripts/run_analysis.sh --skip-install   # do not pip install (deps already present)
#   bash scripts/run_analysis.sh --venv           # isolate deps in ./.venv (recommended on VMs)
#   bash scripts/run_analysis.sh --splits "train_sft test_sft"   # download a subset only
#
# Dependency install is robust to PEP 668 "externally-managed-environment"
# (common on Debian/Ubuntu GCP VMs): it tries a plain install, then falls back
# to --user, then --break-system-packages. Use --venv to sidestep this entirely.
#
# Environment overrides (see pipeline/config.py):
#   UC_DATA_DIR, UC_RESULTS_DIR, UC_HF_REPO
#
set -euo pipefail

# --- locate the repo root (parent of this scripts/ dir) so `python -m` and
#     relative paths resolve regardless of the caller's CWD --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"
SKIP_DOWNLOAD=0
SKIP_INSTALL=0
USE_VENV=0
DOWNLOAD_SPLITS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-install)  SKIP_INSTALL=1; shift ;;
    --venv)          USE_VENV=1; shift ;;
    --splits)        DOWNLOAD_SPLITS="$2"; shift 2 ;;
    -h|--help)       grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

RESULTS_DIR="${UC_RESULTS_DIR:-$ROOT_DIR/results}"
LOG_FILE="$RESULTS_DIR/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$RESULTS_DIR"

log() { echo -e "\n\033[1;34m==> $*\033[0m" | tee -a "$LOG_FILE"; }

# run a stage, mirroring stdout+stderr to the log, timing it
run_stage() {
  local label="$1"; shift
  log "$label"
  local t0 t1
  t0=$(date +%s)
  "$@" 2>&1 | tee -a "$LOG_FILE"
  t1=$(date +%s)
  log "$label completed in $((t1 - t0))s"
}

log "Pipeline start — $(date)"
log "ROOT_DIR=$ROOT_DIR  RESULTS_DIR=$RESULTS_DIR  PYTHON=$PY"

# Install requirements, tolerating PEP 668 externally-managed environments.
# Returns non-zero only if every strategy fails.
install_deps() {
  if "$PY" -m pip install -q -r requirements.txt 2>>"$LOG_FILE"; then
    return 0
  fi
  log "Plain pip install failed (likely externally-managed); retrying with --user"
  if "$PY" -m pip install -q --user -r requirements.txt 2>>"$LOG_FILE"; then
    return 0
  fi
  log "Retrying with --break-system-packages"
  "$PY" -m pip install -q --break-system-packages -r requirements.txt 2>>"$LOG_FILE"
}

# --- 0. dependencies --------------------------------------------------------
if [[ "$USE_VENV" -eq 1 ]]; then
  if [[ ! -d "$ROOT_DIR/.venv" ]]; then
    run_stage "Creating virtualenv (.venv)" "$PY" -m venv "$ROOT_DIR/.venv"
  fi
  PY="$ROOT_DIR/.venv/bin/python"
  log "Using virtualenv interpreter: $PY"
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  log "Installing dependencies"
  if install_deps; then
    log "Dependencies installed"
  else
    log "ERROR: could not install dependencies by any method; see $LOG_FILE"
    exit 1
  fi
else
  log "Skipping dependency install (--skip-install)"
fi

# --- 1. download ------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  if [[ -n "$DOWNLOAD_SPLITS" ]]; then
    # shellcheck disable=SC2086
    run_stage "Stage 0: download dataset (splits: $DOWNLOAD_SPLITS)" \
      "$PY" -m pipeline.analysis.download_dataset --splits $DOWNLOAD_SPLITS
  else
    run_stage "Stage 0: download dataset (all splits)" \
      "$PY" -m pipeline.analysis.download_dataset
  fi
else
  log "Skipping download (--skip-download); using existing UC_DATA_DIR"
fi

# --- 2. analysis stages -----------------------------------------------------
run_stage "Stage 1: structure"          "$PY" -m pipeline.analysis.analyze_structure
run_stage "Stage 2: text volume + cost" "$PY" -m pipeline.analysis.analyze_text_stats
run_stage "Stage 3: content features"   "$PY" -m pipeline.analysis.analyze_content_features
run_stage "Stage 4: unicode / scripts"  "$PY" -m pipeline.analysis.analyze_unicode_scripts
run_stage "Stage 5: translation risks"  "$PY" -m pipeline.analysis.analyze_translation_risks

# --- 3. report --------------------------------------------------------------
run_stage "Stage 6: aggregate report"   "$PY" -m pipeline.analysis.aggregate_report

log "Pipeline complete."
log "Report:  $RESULTS_DIR/ANALYSIS_REPORT.md"
log "Log:     $LOG_FILE"
