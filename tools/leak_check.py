#!/usr/bin/env python3
"""Train/test data-leakage check for the finalized UltraChat-Sinhala splits.

Leakage = a test dialogue (or a duplicate of one) also present in train. Per
dataset (SFT, GEN) it checks:
  * prompt_id overlap train∩test  — same source dialogue on both sides (must be 0)
  * identical-dialogue overlap     — md5 of messages; catches duplicate dialogues
                                     with different prompt_ids split across sides
  * duplicate dialogues within each side (info)
  * shared first-user-turn          — soft signal (same opening question on both
                                     sides; expected for UltraChat, not label leak)
Plus a cross-dataset prompt_id overlap (SFT vs GEN), which should be 0 for global
id-uniqueness.

Usage:
  python3 tools/leak_check.py --base data/final_split
  python3 tools/leak_check.py --base data/final_split --suffix .sinhala.jsonl
"""
import argparse
import hashlib
import json
import os


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load(path):
    pids, chash, uhash = [], [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        pids.append(r["prompt_id"])
        msgs = r["messages"]
        chash.append(md5(json.dumps(msgs, ensure_ascii=False)))
        fu = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        uhash.append(md5(fu.strip()))
    return pids, chash, uhash


def report(name, trf, tef):
    trp, trc, tru = load(trf)
    tep, tec, teu = load(tef)
    trp_s, tep_s, trc_s, tru_s = set(trp), set(tep), set(trc), set(tru)
    pid_ov = len(trp_s & tep_s)
    content_ov = len(set(tec) & trc_s)
    test_leaked = sum(1 for c in tec if c in trc_s)
    uturn_test = sum(1 for u in teu if u in tru_s)
    tag = lambda b: "PASS" if b else "LEAK"
    print(f"\n===== {name} =====")
    print(f"  train={len(trp):,}  test={len(tep):,}")
    print(f"  [{tag(pid_ov==0)}] prompt_id in BOTH train & test : {pid_ov}")
    print(f"  [{tag(test_leaked==0)}] identical dialogue in train    : {content_ov} distinct, {test_leaked} test records")
    print(f"  [info]   dup dialogues within train : {len(trc)-len(trc_s)}")
    print(f"  [info]   dup dialogues within test  : {len(tec)-len(set(tec))}")
    print(f"  [soft]   test records sharing a first-user-turn with train: "
          f"{uturn_test} ({100*uturn_test/max(len(tep),1):.2f}%)")
    return trp_s | tep_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/final_split")
    ap.add_argument("--suffix", default=".sinhala.jsonl")
    a = ap.parse_args()
    f = lambda s: os.path.join(a.base, s + a.suffix)
    sft_all = report("SFT", f("train_sft"), f("test_sft"))
    gen_all = report("GEN", f("train_gen"), f("test_gen"))
    print("\n===== cross-dataset =====")
    print(f"  prompt_ids shared between SFT and GEN: {len(sft_all & gen_all)}")


if __name__ == "__main__":
    main()
