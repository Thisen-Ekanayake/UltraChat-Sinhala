#!/usr/bin/env bash
#
# run_all_parts_mi300x.sh — launch the per-part SFT translation jobs on the
# MI300X, one tmux window per part, optionally capped concurrency.
#
# It first pre-caches the NLLB model once (so the parallel jobs don't race the
# download), then starts the jobs. Each part is independent and resumable, so
# re-running this script just continues where the parts left off.
#
# Usage (from anywhere):
#   bash scripts/run_all_parts_mi300x.sh              # all 10 parts in parallel
#   PARTS=10 CONCURRENCY=4 bash scripts/run_all_parts_mi300x.sh   # 4 at a time
#
# Monitor:   tmux attach -t xl8        (Ctrl-b n / p to switch parts)
# Per-part time lands in results/translate_part_NN_*.log (the final "DONE" line).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
PY="${PYTHON:-python3}"

PARTS="${PARTS:-10}"
CONCURRENCY="${CONCURRENCY:-$PARTS}"     # how many parts to run at once
SESSION="${SESSION:-xl8}"
export UC_NLLB_MODEL="${UC_NLLB_MODEL:-facebook/nllb-200-3.3B}"

command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }

echo "==> Pre-caching $UC_NLLB_MODEL (one-time download so jobs don't race)"
"$PY" - <<PY
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
m = os.environ["UC_NLLB_MODEL"]
AutoTokenizer.from_pretrained(m)
AutoModelForSeq2SeqLM.from_pretrained(m)
print("cached:", m)
PY

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
  exit 1
}

echo "==> Launching $PARTS parts ($CONCURRENCY at a time) in tmux session '$SESSION'"
tmux new-session -d -s "$SESSION" -n control "echo control window; exec bash"

for i in $(seq 1 "$PARTS"); do
  PART=$(printf "part_%02d" "$i")
  # Throttle: before starting part i, wait until running jobs < CONCURRENCY.
  while [[ "$(pgrep -fc 'pipeline.translation.translate run' || echo 0)" -ge "$CONCURRENCY" ]]; do
    sleep 10
  done
  echo "  starting $PART"
  tmux new-window -t "$SESSION" -n "$PART" \
    "bash scripts/run_translate_mi300x.sh $PART; echo; echo '[$PART finished]'; exec bash"
  sleep 3      # small stagger so model loads don't all hit at once
done

echo "==> All parts launched. Attach: tmux attach -t $SESSION"
echo "    Per-part time = final 'DONE' / 'total job time' line in results/translate_part_NN_*.log"
