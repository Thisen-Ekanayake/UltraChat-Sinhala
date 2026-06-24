#!/usr/bin/env bash
#
# upload_watch.sh — watch for completed translated parts of a dataset and upload
# each to a GCS prefix exactly once, as soon as it finishes.
#
# Completion is detected via the translator's "DONE <part>" log line, which is
# only written after the part's jsonl is fully flushed — so a partially-written
# file is never uploaded. Idempotent (per-part marker), retries on failure, and
# exits once every part is uploaded. Survives disconnects when run under nohup.
#
# Env:
#   DEST          gs://bucket/prefix/   (required; trailing slash added if absent)
#   PART_PREFIX   part (SFT, default) | gen (GEN)
#   PARTS         number of parts (default 10)
#   INTERVAL      poll seconds (default 300)
#   GSUTIL        path to gsutil (default /root/google-cloud-sdk/bin/gsutil)
#
# Example:
#   DEST=gs://sinllama-cpt/UltraChat-Sinhala/gen/uncleaned/ PART_PREFIX=gen \
#     nohup bash scripts/upload_watch.sh > upload_gen.log 2>&1 &
#
set -uo pipefail   # NOT -e: a transient gsutil failure must retry, not kill the watcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

DEST="${DEST:?set DEST=gs://bucket/prefix/}"
[[ "$DEST" == */ ]] || DEST="$DEST/"
PREFIX="${PART_PREFIX:-part}"
PARTS="${PARTS:-10}"
INTERVAL="${INTERVAL:-300}"
GS="${GSUTIL:-/root/google-cloud-sdk/bin/gsutil}"
MARKERS="$HOME/.uploaded_parts"
mkdir -p "$MARKERS"

log() { echo "[$(date '+%F %T')] $*"; }
log "uploader: prefix=${PREFIX}_ parts=$PARTS -> $DEST (poll ${INTERVAL}s)"

while :; do
  done_count=0
  for i in $(seq -w 1 "$PARTS"); do
    part="${PREFIX}_$i"
    if [ -f "$MARKERS/$part" ]; then
      done_count=$((done_count + 1)); continue
    fi
    f="data/translated_${part}/${part}.sinhala.jsonl"
    if grep -qa "DONE $part" results/translate_${part}_*.log 2>/dev/null && [ -f "$f" ]; then
      log "$part complete ($(wc -l < "$f") lines, $(du -h "$f" | cut -f1)) -> uploading"
      if "$GS" -m cp "$f" "$DEST"; then
        touch "$MARKERS/$part"
        done_count=$((done_count + 1))
        log "$part uploaded OK"
      else
        log "$part upload FAILED — will retry next cycle"
      fi
    fi
  done
  if [ "$done_count" -ge "$PARTS" ]; then
    log "all $PARTS parts uploaded. exiting."
    break
  fi
  sleep "$INTERVAL"
done
