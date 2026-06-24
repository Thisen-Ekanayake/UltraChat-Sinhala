#!/usr/bin/env bash
#
# adaptive_parts_mi300x.sh — launch parts of one dataset with concurrency that
# adapts to another dataset's load on the shared GPU.
#
# Runs LOW concurrent PART_PREFIX jobs while ANY WATCH_PREFIX job is still on the
# GPU, then ramps to HIGH once they finish — so e.g. GEN runs 2-wide alongside a
# finishing SFT run and automatically goes 4-wide when the GPU frees up. It picks
# the next part that is neither done (translator logged "DONE <part>") nor
# already running, so it never starts a duplicate and is safe to take over from
# a fixed-concurrency orchestrator. Exits when all parts are done.
#
# Env: PART_PREFIX (e.g. gen), WATCH_PREFIX (e.g. part), LOW (2), HIGH (4),
#      PARTS (10), UC_DATA_DIR, UC_TRANSLATE_BATCH.
#
# Example (take over GEN, ramp to 4 once SFT 'part_' jobs end):
#   PART_PREFIX=gen WATCH_PREFIX=part LOW=2 HIGH=4 UC_DATA_DIR=$PWD/data/parts_gen \
#     nohup bash scripts/adaptive_parts_mi300x.sh > gen_adaptive.log 2>&1 &
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PREFIX="${PART_PREFIX:?set PART_PREFIX (e.g. gen)}"
WATCH="${WATCH_PREFIX:-part}"
LOW="${LOW:-2}"
HIGH="${HIGH:-4}"
PARTS="${PARTS:-10}"
export UC_TRANSLATE_BATCH="${UC_TRANSLATE_BATCH:-96}"

log() { echo "[$(date '+%F %T')] $*"; }

part_done()    { grep -qa "DONE ${PREFIX}_$1" results/translate_${PREFIX}_$1_*.log 2>/dev/null; }
part_running() { pgrep -f "translate run --splits ${PREFIX}_$1" >/dev/null 2>&1; }
running_cnt()  { pgrep -fc "translate run --splits ${PREFIX}_"; }
watch_cnt()    { pgrep -fc "translate run --splits ${WATCH}_"; }

log "adaptive: prefix=${PREFIX} watch=${WATCH} low=$LOW high=$HIGH parts=$PARTS batch=$UC_TRANSLATE_BATCH data=${UC_DATA_DIR:-<default>}"
ramped=0
while :; do
  all_done=1
  for i in $(seq -w 1 "$PARTS"); do part_done "$i" || { all_done=0; break; }; done
  if [ "$all_done" -eq 1 ]; then log "all $PARTS ${PREFIX}_ parts done. exiting."; break; fi

  if [ "$(watch_cnt)" -gt 0 ]; then
    target="$LOW"
  else
    target="$HIGH"
    if [ "$ramped" -eq 0 ]; then log "watched '${WATCH}_' jobs finished -> ramping ${PREFIX} concurrency to $HIGH"; ramped=1; fi
  fi

  cur="$(running_cnt)"
  if [ "$cur" -lt "$target" ]; then
    for i in $(seq -w 1 "$PARTS"); do
      if ! part_done "$i" && ! part_running "$i"; then
        log "launch ${PREFIX}_$i  (running=$cur target=$target, ${WATCH}=$(watch_cnt))"
        nohup bash "$SCRIPT_DIR/run_translate_mi300x.sh" "${PREFIX}_$i" >/dev/null 2>&1 &
        sleep 20      # let it register before the next slot decision
        break
      fi
    done
  fi
  sleep 20
done
