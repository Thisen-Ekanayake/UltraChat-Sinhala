#!/usr/bin/env python3
"""Stage 3 — prevalence of MT-sensitive content features.

For every message, runs the shared detectors (code, inline code, URLs, tables,
lists, headings, LaTeX/math, HTML, emoji, links, emphasis) and tallies how many
messages contain each feature, per split and dataset-wide. Reports both raw
counts and percentage of messages.

This quantifies *how much* of the corpus needs special handling (masking /
structure preservation) before translation. Output: results/03_content_features.json
"""
from __future__ import annotations

import sys
from collections import Counter

from pipeline import config
from pipeline.detectors import DETECTORS
from pipeline.common import (
    StepTimer, fmt_int, get_logger, iter_messages, iter_split_records,
    require_splits, save_result,
)

log = get_logger("content")
OUTPUT = "03_content_features.json"


def analyze_split(split: str, shards) -> dict:
    counts = Counter()
    n_messages = 0
    n_with_any = 0
    for rec in iter_split_records(split, shards):
        for m in iter_messages(rec):
            n_messages += 1
            hit_any = False
            text = m["content"]
            for name, pat in DETECTORS.items():
                if pat.search(text):
                    counts[name] += 1
                    hit_any = True
            if hit_any:
                n_with_any += 1
    return {
        "n_messages": n_messages,
        "n_messages_with_any_feature": n_with_any,
        "feature_counts": {name: counts.get(name, 0) for name in DETECTORS},
        "feature_pct": {
            name: round(100 * counts.get(name, 0) / n_messages, 3) if n_messages else 0.0
            for name in DETECTORS
        },
    }


def main() -> int:
    splits = require_splits(log)
    per_split = {}
    for split in config.ordered_splits():
        with StepTimer(log, f"content[{split}]"):
            per_split[split] = analyze_split(split, splits[split])
            top = sorted(per_split[split]["feature_counts"].items(),
                         key=lambda kv: -kv[1])[:3]
            log.info("  %s: top features %s",
                     split, ", ".join(f"{k}={fmt_int(v)}" for k, v in top))

    n_messages = sum(s["n_messages"] for s in per_split.values())
    agg = Counter()
    agg_any = 0
    for s in per_split.values():
        agg_any += s["n_messages_with_any_feature"]
        for name, c in s["feature_counts"].items():
            agg[name] += c
    overall = {
        "n_messages": n_messages,
        "n_messages_with_any_feature": agg_any,
        "feature_counts": dict(agg),
        "feature_pct": {
            name: round(100 * agg[name] / n_messages, 3) if n_messages else 0.0
            for name in DETECTORS
        },
    }
    save_result(OUTPUT, {"per_split": per_split, "overall": overall})
    log.info("Wrote %s", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
