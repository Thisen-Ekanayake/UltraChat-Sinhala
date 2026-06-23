#!/usr/bin/env python3
"""Stage 5 — Sinhala machine-translation risk register.

Synthesises the prevalence measurements from stages 3 and 4 into a prioritised
risk register for an English->Sinhala translation pipeline, and performs one
additional streaming pass to:
  * count messages carrying at least one HIGH-severity feature (these require
    placeholder masking before translation),
  * count messages exceeding chunking thresholds (long-message risk), and
  * collect short, truncated example snippets per risk category for the report.

The output is the document that directly answers "what could go wrong when
translating this corpus to Sinhala". Output: results/05_translation_risks.json

Note on examples: snippets are the first matches encountered (illustrative, not
a random sample) and are truncated to EXAMPLE_MAXLEN characters with newlines
collapsed, purely for readability in the report.
"""
from __future__ import annotations

import sys
from collections import Counter

from pipeline import config
from pipeline.detectors import DETECTORS, RATIONALE, SEVERITY
from pipeline.common import (
    StepTimer, fmt_int, get_logger, iter_messages, iter_split_records,
    load_result, require_splits, save_result,
)

log = get_logger("risks")
OUTPUT = "05_translation_risks.json"

HIGH_SEVERITY = [k for k, v in SEVERITY.items() if v == "high"]
EXAMPLES_PER_CATEGORY = 3
EXAMPLE_MAXLEN = 220
# Messages longer than this likely need sentence-aware chunking before MT.
CHUNK_THRESHOLDS = [5000, 10000, 30000]


def _snippet(text: str) -> str:
    s = " ".join(text.split())
    return s[:EXAMPLE_MAXLEN] + ("…" if len(s) > EXAMPLE_MAXLEN else "")


def collect_examples_and_flags(splits) -> dict:
    examples: dict[str, list[str]] = {name: [] for name in DETECTORS}
    n_messages = 0
    n_high = 0
    over = {t: 0 for t in CHUNK_THRESHOLDS}
    max_len = 0

    for split in config.ordered_splits():
        with StepTimer(log, f"risks[{split}]"):
            for rec in iter_split_records(split, splits[split]):
                for m in iter_messages(rec):
                    n_messages += 1
                    text = m["content"]
                    clen = len(text)
                    max_len = max(max_len, clen)
                    for t in CHUNK_THRESHOLDS:
                        if clen > t:
                            over[t] += 1
                    has_high = False
                    for name, pat in DETECTORS.items():
                        if len(examples[name]) < EXAMPLES_PER_CATEGORY or SEVERITY[name] == "high":
                            if pat.search(text):
                                if len(examples[name]) < EXAMPLES_PER_CATEGORY:
                                    examples[name].append(_snippet(text))
                                if SEVERITY[name] == "high":
                                    has_high = True
                        elif SEVERITY[name] == "high" and pat.search(text):
                            has_high = True
                    if has_high:
                        n_high += 1

    return {
        "n_messages": n_messages,
        "n_messages_high_severity": n_high,
        "pct_messages_high_severity": round(100 * n_high / n_messages, 3) if n_messages else 0.0,
        "messages_over_length": {str(t): over[t] for t in CHUNK_THRESHOLDS},
        "max_message_chars": max_len,
        "examples": examples,
    }


def build_register(pass_data: dict) -> list[dict]:
    """Merge prevalence (stage 3) + scripts (stage 4) into a ranked register."""
    try:
        content = load_result("03_content_features.json")["overall"]
        feat_counts = content["feature_counts"]
        feat_pct = content["feature_pct"]
    except Exception:
        log.warning("03_content_features.json not found; counts omitted.")
        feat_counts, feat_pct = {}, {}

    sev_rank = {"high": 0, "medium": 1, "low": 2}
    register = []
    for name in DETECTORS:
        register.append({
            "category": name,
            "severity": SEVERITY[name],
            "rationale": RATIONALE[name],
            "messages_affected": feat_counts.get(name, 0),
            "pct_messages": feat_pct.get(name, 0.0),
            "examples": pass_data["examples"].get(name, []),
        })
    register.sort(key=lambda r: (sev_rank[r["severity"]], -r["messages_affected"]))
    return register


def foreign_script_risk() -> dict:
    try:
        u = load_result("04_unicode_scripts.json")["overall"]
    except Exception:
        return {}
    scripts = u.get("script_message_counts", {})
    sinhala_already = scripts.get("Sinhala", 0)
    return {
        "n_foreign_script_messages": u.get("n_foreign_script_messages", 0),
        "foreign_script_chars": u.get("foreign_script_chars", 0),
        "messages_already_in_sinhala": sinhala_already,
        "top_foreign_scripts": dict(list(scripts.items())[:10]),
        "note": ("Foreign-script source text degrades en->si MT; messages "
                 "already containing Sinhala may be double-translated."),
    }


def main() -> int:
    splits = require_splits(log)
    pass_data = collect_examples_and_flags(splits)
    register = build_register(pass_data)

    summary = {
        "n_messages": pass_data["n_messages"],
        "n_messages_high_severity": pass_data["n_messages_high_severity"],
        "pct_messages_high_severity": pass_data["pct_messages_high_severity"],
        "messages_over_length": pass_data["messages_over_length"],
        "max_message_chars": pass_data["max_message_chars"],
        "high_severity_categories": HIGH_SEVERITY,
        "chunk_thresholds": CHUNK_THRESHOLDS,
    }
    save_result(OUTPUT, {
        "summary": summary,
        "risk_register": register,
        "foreign_script_risk": foreign_script_risk(),
    })
    log.info("Wrote %s (%s msgs need masking, %.2f%%)",
             OUTPUT, fmt_int(summary["n_messages_high_severity"]),
             summary["pct_messages_high_severity"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
