#!/usr/bin/env bash
#
# upload_watch.sh — sync translated parts (partial AND complete) to a GCS prefix
# and publish a resume manifest, so translation work survives a pod shutdown and
# can be continued later without re-translating anything.
#
# Every cycle it:
#   1. Uploads each part's .sinhala.jsonl — partials are re-synced (overwrite)
#      while they grow; a part that hits the translator's "DONE <part>" marker is
#      uploaded one final time and then marked so it is not re-uploaded.
#   2. Writes results/<prefix>_progress.{json,txt} — per-part lines/total/pct/
#      status (done|partial|queued) + a resume_from_line, plus total dialogues
#      remaining and an MI300X ETA (from the recently observed aggregate rate),
#      and uploads them alongside the data.
#
# To continue later: download each "partial" part's .sinhala.jsonl back to
# data/translated_<part>/ and re-run the translator — it skips prompt_ids already
# present and resumes from resume_from_line. The manifest tells you which parts
# are partial and how far they got.
#
# set -uo pipefail (NOT -e): a transient gsutil failure must retry, not kill it.
#
# Env:
#   DEST          gs://bucket/prefix/   (required; trailing slash added if absent)
#   PART_PREFIX   gen (default) | part
#   PARTS         number of parts (default 10)
#   INTERVAL      poll seconds (default 600)
#   FALLBACK_RATE aggregate dialogues/hour used for ETA before a real rate is
#                 measured (default 11000 — observed 4-wide MI300X throughput)
#   GSUTIL        path to gsutil (default /root/google-cloud-sdk/bin/gsutil)
#
# Example:
#   DEST=gs://sinllama-cpt/UltraChat-Sinhala/gen/uncleaned/ PART_PREFIX=gen \
#     nohup bash scripts/upload_watch.sh > upload_gen.log 2>&1 &
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

DEST="${DEST:?set DEST=gs://bucket/prefix/}"
[[ "$DEST" == */ ]] || DEST="$DEST/"
export PART_PREFIX="${PART_PREFIX:-gen}"
export PARTS="${PARTS:-10}"
export FALLBACK_RATE="${FALLBACK_RATE:-11000}"
PREFIX="$PART_PREFIX"
INTERVAL="${INTERVAL:-600}"
GS="${GSUTIL:-/root/google-cloud-sdk/bin/gsutil}"
MARKERS="$HOME/.uploaded_parts"
mkdir -p "$MARKERS" results

log() { echo "[$(date '+%F %T')] $*"; }
log "uploader: prefix=${PREFIX}_ parts=$PARTS -> $DEST (partial sync, poll ${INTERVAL}s)"

# Build + upload the resume manifest (per-part progress, remaining, MI300X ETA).
write_manifest() {
  python3 - <<'PYEOF'
import json, os, glob, time

ROOT   = os.getcwd()
PREFIX = os.environ.get("PART_PREFIX", "gen")
PARTS  = int(os.environ.get("PARTS", "10"))
RATE0  = float(os.environ.get("FALLBACK_RATE", "11000"))   # dlg/h aggregate
STATE  = os.path.join(ROOT, "results", f".{PREFIX}_rate_state.json")
OJSON  = os.path.join(ROOT, "results", f"{PREFIX}_progress.json")
OTXT   = os.path.join(ROOT, "results", f"{PREFIX}_progress.txt")

# per-part dialogue totals from the split manifest
totals = {}
for p in (f"data/parts_{PREFIX}/parts_manifest.json",
          "data/parts/parts_manifest.json",
          "data/parts_manifest.json"):
    if os.path.exists(p):
        pp = json.load(open(p)); pp = pp.get("per_part", pp)
        for k, v in pp.items():
            if k.startswith(PREFIX + "_") and isinstance(v, dict):
                totals[k] = v.get("dialogues")
        if totals:
            break

def is_done(part):
    for f in glob.glob(f"results/translate_{part}_*.log"):
        try:
            with open(f, "rb") as fh:
                if f"DONE {part}".encode() in fh.read():
                    return True
        except OSError:
            pass
    return False

def nlines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return n

rows, sum_done, sum_total = [], 0, 0
for i in range(1, PARTS + 1):
    part  = f"{PREFIX}_{i:02d}"
    total = totals.get(part) or 28433
    f     = f"data/translated_{part}/{part}.sinhala.jsonl"
    if is_done(part):
        cnt, status = total, "done"
    else:
        cnt = nlines(f)
        status = "partial" if cnt > 0 else "queued"
    cnt = min(cnt, total)
    rows.append((part, cnt, total, status))
    sum_done  += cnt
    sum_total += total

remaining = sum_total - sum_done

# empirical aggregate throughput from the delta since the previous cycle
now, rate, prev = time.time(), RATE0, None
if os.path.exists(STATE):
    try:
        prev = json.load(open(STATE))
    except (OSError, ValueError):
        prev = None
if prev and now > prev.get("ts", 0):
    d_done = sum_done - prev.get("done", 0)
    d_t    = now - prev["ts"]
    if d_done > 0 and d_t > 0:
        rate = d_done / d_t * 3600.0
json.dump({"ts": now, "done": sum_done}, open(STATE, "w"))

eta_h = remaining / rate if rate > 0 else float("inf")

manifest = {
    "prefix": PREFIX,
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "parts": [
        {"part": p, "done": c, "total": t, "pct": round(100 * c / t, 1),
         "status": s, "resume_from_line": (c if s == "partial" else None)}
        for (p, c, t, s) in rows
    ],
    "dialogues_done": sum_done,
    "dialogues_total": sum_total,
    "dialogues_remaining": remaining,
    "aggregate_rate_dlg_per_h": round(rate),
    "eta_hours_mi300x": round(eta_h, 2),
}
json.dump(manifest, open(OJSON, "w"), indent=2)

with open(OTXT, "w") as fh:
    fh.write(f"{PREFIX.upper()} translation progress — {manifest['generated']}\n\n")
    fh.write(f"{'part':10} {'done':>8} {'total':>8} {'pct':>6}  status\n")
    fh.write("-" * 46 + "\n")
    for (p, c, t, s) in rows:
        fh.write(f"{p:10} {c:8d} {t:8d} {100 * c / t:5.1f}%  {s}\n")
    fh.write("-" * 46 + "\n")
    fh.write(f"Dialogues done : {sum_done:,} / {sum_total:,}\n")
    fh.write(f"Remaining      : {remaining:,}\n")
    fh.write(f"Aggregate rate : {round(rate):,} dlg/h  (MI300X, recently observed)\n")
    fh.write(f"ETA to finish  : {eta_h:.2f} h\n\n")
    fh.write("Resume: download each 'partial' part's .sinhala.jsonl back to\n")
    fh.write("data/translated_<part>/ and re-run the translator; it skips prompt_ids\n")
    fh.write("already present and continues from resume_from_line.\n")

print(f"manifest: done={sum_done} remaining={remaining} "
      f"rate={round(rate)}/h eta={eta_h:.2f}h")
PYEOF
}

while :; do
  all_done=1
  for i in $(seq -w 1 "$PARTS"); do
    part="${PREFIX}_$i"
    f="data/translated_${part}/${part}.sinhala.jsonl"
    [ -f "$MARKERS/$part" ] && continue   # complete + final-uploaded already

    if grep -qa "DONE $part" results/translate_${part}_*.log 2>/dev/null && [ -f "$f" ]; then
      log "$part COMPLETE ($(wc -l < "$f") lines, $(du -h "$f" | cut -f1)) -> final upload"
      if "$GS" -m cp "$f" "$DEST"; then
        touch "$MARKERS/$part"; log "$part uploaded (final)"
      else
        log "$part final upload FAILED — retry next cycle"; all_done=0
      fi
    elif [ -s "$f" ]; then
      log "$part partial ($(wc -l < "$f") lines, $(du -h "$f" | cut -f1)) -> sync"
      "$GS" -m cp "$f" "$DEST" || log "$part partial sync FAILED — retry next cycle"
      all_done=0
    else
      all_done=0   # queued / not started
    fi
  done

  if write_manifest; then
    "$GS" -m cp "results/${PREFIX}_progress.json" "results/${PREFIX}_progress.txt" "$DEST" \
      || log "manifest upload FAILED — retry next cycle"
  else
    log "manifest build FAILED — skipping manifest upload this cycle"
  fi

  if [ "$all_done" -eq 1 ]; then
    log "all $PARTS parts complete + uploaded. exiting."
    break
  fi
  sleep "$INTERVAL"
done
