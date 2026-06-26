#!/usr/bin/env python3
"""Partition cleaned UltraChat-Sinhala shards into train/test by prompt_id,
mirroring the original HuggingFaceH4/ultrachat_200k split exactly.

The round-robin shards mix train and test; membership is recovered by joining on
prompt_id to the original split. Test = the original test split's prompt_ids;
train = everything else. Raw jsonl lines are copied verbatim (no re-serialisation).

Usage:
  python3 tools/make_train_test.py --in 'data/final/part_*.sinhala.jsonl' \
      --test-ids test_sft_ids.txt \
      --train-out out/train_sft.sinhala.jsonl --test-out out/test_sft.sinhala.jsonl
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inglob", required=True)
    ap.add_argument("--test-ids", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--test-out", required=True)
    a = ap.parse_args()

    test_ids = {ln.strip() for ln in open(a.test_ids) if ln.strip()}
    for p in (a.train_out, a.test_out):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    n_tr = n_te = 0
    seen_test = set()
    with open(a.train_out, "w", encoding="utf-8") as tr, \
         open(a.test_out, "w", encoding="utf-8") as te:
        for f in sorted(glob.glob(a.inglob)):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                pid = json.loads(line).get("prompt_id")
                if not line.endswith("\n"):
                    line += "\n"
                if pid in test_ids:
                    te.write(line); n_te += 1; seen_test.add(pid)
                else:
                    tr.write(line); n_tr += 1
    print(f"  test_ids supplied : {len(test_ids):,}")
    print(f"  train -> {a.train_out}  : {n_tr:,}")
    print(f"  test  -> {a.test_out}  : {n_te:,}  (test ids present in data: {len(seen_test):,})")


if __name__ == "__main__":
    main()
