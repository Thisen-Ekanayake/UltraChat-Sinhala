#!/usr/bin/env bash
#
# run_all_parts_mi300x.sh — launch the per-part SFT translation jobs on the
# MI300X with throttled concurrency.
#
# Empirical note: a single MI300X is compute-bound and saturates at ~4
# concurrent NLLB-3.3B processes (≈93% GPU util). Going higher does NOT add
# throughput and fills the 192 GB VRAM (PyTorch's ROCm caching allocator
# reserves aggressively), which OOMs jobs. So the default concurrency is 4 and
# the per-process batch is 96 — measured stable, ~4–4.5k src-char/s per part,
# ~16–18k aggregate, ~8–9h per part, ~21h for the whole SFT split.
#
# Each part is independent and resumable, so re-running continues where it left
# off. Run it in the background so it survives disconnect:
#   nohup bash scripts/run_all_parts_mi300x.sh > driver.log 2>&1 &
#
# Tune: PARTS, CONCURRENCY, UC_TRANSLATE_BATCH.
# Per-part time = final "DONE … in <time>" line in results/translate_part_NN_*.log.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
PY="${PYTHON:-python3}"

PARTS="${PARTS:-10}"
CONCURRENCY="${CONCURRENCY:-4}"
export UC_NLLB_MODEL="${UC_NLLB_MODEL:-facebook/nllb-200-3.3B}"
export UC_TRANSLATE_BATCH="${UC_TRANSLATE_BATCH:-96}"
export UC_DIALOGUE_CHUNK="${UC_DIALOGUE_CHUNK:-128}"

echo "==> Pre-caching $UC_NLLB_MODEL (one-time, so parallel jobs don't race the download)"
"$PY" - <<'PY'
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
m = os.environ["UC_NLLB_MODEL"]
AutoTokenizer.from_pretrained(m)
AutoModelForSeq2SeqLM.from_pretrained(m)
print("cached:", m)
PY

echo "==> Launching $PARTS parts, $CONCURRENCY at a time, batch=$UC_TRANSLATE_BATCH"
for i in $(seq -w 1 "$PARTS"); do
  # Wait for a free slot (count running translate jobs).
  while [ "$(pgrep -fc 'translate run')" -ge "$CONCURRENCY" ]; do sleep 10; done
  echo "  launch part_$i  $(date '+%H:%M:%S')"
  nohup bash "$SCRIPT_DIR/run_translate_mi300x.sh" "part_$i" >/dev/null 2>&1 &
  sleep 15      # let this job's model load + register before checking the next slot
done
wait
echo "==> All $PARTS parts finished  $(date '+%H:%M:%S')"
