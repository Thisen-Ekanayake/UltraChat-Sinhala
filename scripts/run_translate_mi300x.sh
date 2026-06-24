#!/usr/bin/env bash
#
# run_translate_mi300x.sh — translate ONE part of the split SFT corpus on an
# AMD MI300X (ROCm/HIP). Each part is an independent, resumable job; with 192 GB
# HBM you can run several parts concurrently (see run_all_parts_mi300x.sh).
#
# Pre-req: the parts exist (scripts produce them via the splitter):
#   python -m pipeline.analysis.download_dataset --splits train_sft test_sft
#   python -m pipeline.translation.split_dataset --parts 10
#
# Usage (from anywhere):
#   bash scripts/run_translate_mi300x.sh part_01
#   bash scripts/run_translate_mi300x.sh part_02 --limit 200   # quick check
#
# Part data is read from UC_DATA_DIR (default data/parts) and the Sinhala output
# is written to data/translated_<part>/<part>.sinhala.jsonl (resumable). Each job
# prints a live job line and a final "DONE / total job time" — that is the
# translation time for the part.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
PY="${PYTHON:-python3}"

PART="${1:-}"
if [[ -z "$PART" ]]; then
  echo "usage: $(basename "$0") part_NN [extra translate args]" >&2
  exit 1
fi
shift || true

# --- ROCm / MI300X runtime tuning ------------------------------------------
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

# --- translation knobs (512-token seq length enforced in code) -------------
export UC_NLLB_MODEL="${UC_NLLB_MODEL:-facebook/nllb-200-3.3B}"
export UC_DEVICE="${UC_DEVICE:-cuda}"           # ROCm presents as 'cuda' to torch
export UC_DTYPE="${UC_DTYPE:-bf16}"             # native on MI300X
export UC_TRANSLATE_BATCH="${UC_TRANSLATE_BATCH:-128}"   # per-process; lower if running many parts at once
export UC_DIALOGUE_CHUNK="${UC_DIALOGUE_CHUNK:-128}"
export UC_MAX_SEGMENT_CHARS="${UC_MAX_SEGMENT_CHARS:-1600}"
export UC_MAX_INPUT_TOKENS="${UC_MAX_INPUT_TOKENS:-512}"
export UC_NUM_BEAMS="${UC_NUM_BEAMS:-1}"

# --- Sinhala ZWJ restoration: pin the lexicon cache + vocab to stable repo-root
#     paths so they resolve regardless of UC_DATA_DIR (which varies per dataset,
#     e.g. data/parts vs data/parts_gen). NLLB strips the conjunct joiner at
#     encode time; this restores it on decode (pipeline/translation/sinhala_normalize).
export UC_ZWJ_LEXICON_CACHE="${UC_ZWJ_LEXICON_CACHE:-$ROOT_DIR/data/sinhala_lexicon.pkl}"
export UC_ZWJ_VOCAB="${UC_ZWJ_VOCAB:-$ROOT_DIR/tokenizer/unigram_32000_0.9995.vocab}"

# --- I/O: read the part from the parts dir, write to its own output dir -----
# PART encodes the dataset (part_NN for SFT, gen_NN for GEN), so the output dir
# and log name are automatically distinct per dataset.
export UC_DATA_DIR="${UC_DATA_DIR:-$ROOT_DIR/data/parts}"
export UC_OUTPUT_DIR="${UC_OUTPUT_DIR:-$ROOT_DIR/data/translated_${PART}}"

LOG_DIR="${UC_RESULTS_DIR:-$ROOT_DIR/results}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/translate_${PART}_$(date +%Y%m%d_%H%M%S).log"

echo "[$PART] model=$UC_NLLB_MODEL dtype=$UC_DTYPE batch=$UC_TRANSLATE_BATCH"
echo "[$PART] data=$UC_DATA_DIR  out=$UC_OUTPUT_DIR  log=$LOG"
"$PY" -m pipeline.translation.translate run --splits "$PART" "$@" 2>&1 | tee "$LOG"
