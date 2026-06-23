#!/usr/bin/env bash
#
# run_translate.sh — thin wrapper around the NLLB English->Sinhala translation
# stage (pipeline.translation.translate). Runs from the repo root so the
# `pipeline` package is importable, and forwards all arguments through.
#
# Usage (from anywhere):
#   bash scripts/run_translate.sh estimate --chars-per-sec 150 300 600 1000
#   bash scripts/run_translate.sh bench --split test_sft --n 50
#   bash scripts/run_translate.sh run --splits test_sft --limit 20
#   bash scripts/run_translate.sh run --download --splits train_sft test_sft
#
# Mimic the 4-vCPU CPU VM when benchmarking on a bigger host:
#   UC_DEVICE=cpu UC_CPU_THREADS=4 bash scripts/run_translate.sh bench --n 50
#
# Config knobs (env, see pipeline/config.py): UC_NLLB_MODEL, UC_DEVICE, UC_DTYPE,
# UC_CPU_THREADS, UC_TRANSLATE_BATCH, UC_NUM_BEAMS, UC_OUTPUT_DIR, UC_MASK.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"
exec "$PY" -m pipeline.translation.translate "$@"
