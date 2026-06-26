#!/usr/bin/env python3
"""Filter cleaned UltraChat-Sinhala shards: drop dialogues with empty turns and
deduplicate prompt_ids (keep the first kept occurrence within a split).

A turn cannot be excised from a multi-turn dialogue without breaking the
user/assistant alternation — most empty turns sit mid-conversation — so any
dialogue containing an empty / whitespace-only turn is dropped whole. prompt_ids
are made unique by keeping the first non-dropped occurrence; later collisions are
dropped (the UltraChat GEN split has one prompt_id reused across two distinct
dialogues).

Process a whole split at once (all its shards, in order) so cross-shard
duplicates are caught.

Usage:
  python3 tools/filter_dataset.py --in 'data/fixed/part_*.sinhala.jsonl' --outdir data/final
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inglob", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(args.inglob))
    seen = set()
    tot = kept = d_empty = d_dup = 0
    for f in files:
        outp = os.path.join(args.outdir, os.path.basename(f))
        with open(outp, "w", encoding="utf-8") as out:
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                tot += 1
                pid = rec.get("prompt_id")
                if pid in seen:
                    d_dup += 1
                    continue
                if any(not m.get("content", "").strip() for m in rec.get("messages", [])):
                    d_empty += 1
                    continue
                seen.add(pid)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    print(f"{args.inglob}")
    print(f"  total={tot:,}  kept={kept:,}  dropped_empty_turn={d_empty:,}  dropped_dup_pid={d_dup:,}")


if __name__ == "__main__":
    main()
