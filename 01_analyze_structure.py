#!/usr/bin/env python3
"""Stage 1 — conversational structure analysis.

Per split and dataset-wide, computes:
  * record (dialogue) and message counts
  * turns-per-dialogue distribution
  * role inventory and per-role message counts
  * first-message role distribution
  * strict-alternation validity (user/assistant/user/...)
  * count of empty (zero-length) messages
  * schema integrity (records missing required fields)

These properties determine how the translation pipeline must reassemble
dialogues (preserving role order and prompt_id) and flag structural anomalies
that would corrupt an SFT dataset.

Output: results/01_structure.json
"""
from __future__ import annotations

import sys
from collections import Counter

import config
from uc_common import (
    LengthAccumulator, StepTimer, fmt_int, get_logger,
    iter_messages, iter_split_records, require_splits, save_result,
)

log = get_logger("structure")
OUTPUT = "01_structure.json"


def analyze_split(split: str, shards) -> dict:
    turns = LengthAccumulator()
    roles = Counter()
    first_role = Counter()
    n_records = 0
    n_messages = 0
    n_empty = 0
    n_nonalternating = 0
    n_missing_fields = 0
    n_no_prompt_id = 0

    for rec in iter_split_records(split, shards):
        n_records += 1
        if "messages" not in rec or "prompt" not in rec:
            n_missing_fields += 1
        if not rec.get("prompt_id"):
            n_no_prompt_id += 1

        msgs = list(iter_messages(rec))
        turns.add(len(msgs))
        if msgs:
            first_role[msgs[0]["role"]] += 1
        seq = []
        for m in msgs:
            n_messages += 1
            roles[m["role"]] += 1
            if len(m["content"]) == 0:
                n_empty += 1
            seq.append(m["role"])
        if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
            n_nonalternating += 1

    return {
        "n_records": n_records,
        "n_messages": n_messages,
        "turns_per_dialogue": turns.summary(),
        "roles": dict(roles),
        "first_message_role": dict(first_role),
        "n_empty_messages": n_empty,
        "n_nonalternating_dialogues": n_nonalternating,
        "n_records_missing_required_fields": n_missing_fields,
        "n_records_missing_prompt_id": n_no_prompt_id,
    }


def main() -> int:
    splits = require_splits(log)
    per_split = {}
    for split in config.ordered_splits():
        with StepTimer(log, f"structure[{split}]"):
            per_split[split] = analyze_split(split, splits[split])
            s = per_split[split]
            log.info("  %s: %s dialogues, %s messages, median %.0f turns",
                     split, fmt_int(s["n_records"]), fmt_int(s["n_messages"]),
                     s["turns_per_dialogue"]["p50"])

    overall = {
        "n_records": sum(s["n_records"] for s in per_split.values()),
        "n_messages": sum(s["n_messages"] for s in per_split.values()),
        "n_empty_messages": sum(s["n_empty_messages"] for s in per_split.values()),
        "n_nonalternating_dialogues": sum(
            s["n_nonalternating_dialogues"] for s in per_split.values()),
    }
    save_result(OUTPUT, {"per_split": per_split, "overall": overall})
    log.info("Wrote %s", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
