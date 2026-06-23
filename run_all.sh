#!/usr/bin/env bash
#
# run_all.sh — orchestrate the UltraChat 200k analysis pipeline end-to-end.
#
# Runs each stage in order, timing each, and writes the final academic report to
# $UC_RESULTS_DIR/ANALYSIS_REPORT.md. Any stage failure aborts the run.
#
# Usage:
#   bash run_all.sh                  # install deps, download, analyse, report
#   bash run_all.sh --skip-download  # reuse parquet shards already in UC_DATA_DIR
#   bash run_all.sh --skip-install   # do not pip install (deps already present)
#   bash run_all.sh --splits "train_sft test_sft"   # download a subset only
#
# Environment overrides (see config.py):
#   UC_DATA_DIR, UC_RESULTS_DIR, UC_HF_REPO
#
set -euo pipefail

# --- locate ourselves so the script is runnable from any CWD ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-python3}"
SKIP_DOWNLOAD=0
SKIP_INSTALL=0
DOWNLOAD_SPLITS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-install)  SKIP_INSTALL=1; shift ;;
    --splits)        DOWNLOAD_SPLITS="$2"; shift 2 ;;
    -h|--help)       grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

RESULTS_DIR="${UC_RESULTS_DIR:-$SCRIPT_DIR/results}"
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
log "SCRIPT_DIR=$SCRIPT_DIR  RESULTS_DIR=$RESULTS_DIR  PYTHON=$PY"

# --- 0. dependencies --------------------------------------------------------
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  run_stage "Installing dependencies" \
    "$PY" -m pip install -q -r requirements.txt
else
  log "Skipping dependency install (--skip-install)"
fi

# --- 1. download ------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  if [[ -n "$DOWNLOAD_SPLITS" ]]; then
    # shellcheck disable=SC2086
    run_stage "Stage 0: download dataset (splits: $DOWNLOAD_SPLITS)" \
      "$PY" 00_download_dataset.py --splits $DOWNLOAD_SPLITS
  else
    run_stage "Stage 0: download dataset (all splits)" \
      "$PY" 00_download_dataset.py
  fi
else
  log "Skipping download (--skip-download); using existing UC_DATA_DIR"
fi

# --- 2. analysis stages -----------------------------------------------------
run_stage "Stage 1: structure"          "$PY" 01_analyze_structure.py
run_stage "Stage 2: text volume + cost" "$PY" 02_analyze_text_stats.py
run_stage "Stage 3: content features"   "$PY" 03_analyze_content_features.py
run_stage "Stage 4: unicode / scripts"  "$PY" 04_analyze_unicode_scripts.py
run_stage "Stage 5: translation risks"  "$PY" 05_analyze_translation_risks.py

# --- 3. report --------------------------------------------------------------
run_stage "Stage 6: aggregate report"   "$PY" 06_aggregate_report.py

log "Pipeline complete."
log "Report:  $RESULTS_DIR/ANALYSIS_REPORT.md"
log "Log:     $LOG_FILE"
