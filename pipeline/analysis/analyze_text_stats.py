#!/usr/bin/env python3
"""Stage 2 — text volume, length distributions and translation-cost model.

Per split and dataset-wide, computes:
  * message-level and dialogue-level character-length distributions
  * total translatable character volume (what character-billed MT APIs charge on)
  * per-role character volume (user vs assistant)
  * approximate token counts (tiktoken cl100k_base if available, else chars/4)
  * count of messages exceeding common single-request size limits

It then derives a translation-cost model for:
  * one-way (en->si) via Google Cloud Translation
  * round-trip (en->si->en) as proposed for back-translation QE
and compares against the GCP free-trial credit.

Character counts are exact; token counts and costs are clearly labelled
estimates. Output: results/02_text_stats.json
"""
from __future__ import annotations

import sys

from pipeline import config
from pipeline.common import (
    LengthAccumulator, StepTimer, fmt_int, get_logger,
    iter_messages, iter_split_records, require_splits, save_result,
)

log = get_logger("text_stats")
OUTPUT = "02_text_stats.json"

# Practical single-request character ceilings to flag long messages against.
REQUEST_LIMITS = [5000, 30000]

# Try a real tokenizer; fall back to a heuristic if unavailable.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))

    TOKEN_METHOD = "tiktoken/cl100k_base"
except Exception:  # pragma: no cover - heuristic path
    def count_tokens(text: str) -> int:
        return int(round(len(text) / config.CHARS_PER_TOKEN_HEURISTIC))

    TOKEN_METHOD = f"heuristic chars/{config.CHARS_PER_TOKEN_HEURISTIC:g}"


def analyze_split(split: str, shards) -> dict:
    msg_chars = LengthAccumulator()
    dlg_chars = LengthAccumulator()
    chars_by_role = {"user": 0, "assistant": 0, "_other": 0}
    total_tokens = 0
    over_limit = {lim: 0 for lim in REQUEST_LIMITS}

    for rec in iter_split_records(split, shards):
        dialogue_len = 0
        for m in iter_messages(rec):
            clen = len(m["content"])
            msg_chars.add(clen)
            dialogue_len += clen
            bucket = m["role"] if m["role"] in chars_by_role else "_other"
            chars_by_role[bucket] += clen
            total_tokens += count_tokens(m["content"])
            for lim in REQUEST_LIMITS:
                if clen > lim:
                    over_limit[lim] += 1
        dlg_chars.add(dialogue_len)

    return {
        "message_chars": msg_chars.summary(),
        "dialogue_chars": dlg_chars.summary(),
        "total_chars": msg_chars.summary()["sum"],
        "chars_by_role": chars_by_role,
        "approx_total_tokens": total_tokens,
        "messages_over_request_limit": {str(k): v for k, v in over_limit.items()},
    }


def cost_model(total_chars_by_split: dict[str, int]) -> dict:
    """Derive one-way and round-trip MT cost figures from character volumes."""
    rate = config.GOOGLE_TRANSLATE_USD_PER_MCHAR
    credit = config.GCP_FREE_TRIAL_CREDIT_USD
    exp = config.SINHALA_EXPANSION_FACTOR

    def block(chars: int) -> dict:
        one_way = chars / 1e6 * rate
        # round trip bills forward (en chars) + backward (si chars ~ exp*en)
        round_trip = (chars + chars * exp) / 1e6 * rate
        return {
            "chars": chars,
            "one_way_usd": round(one_way, 2),
            "round_trip_usd": round(round_trip, 2),
            "one_way_pct_of_credit": round(100 * one_way / credit, 1),
            "round_trip_pct_of_credit": round(100 * round_trip / credit, 1),
        }

    sft_chars = sum(total_chars_by_split.get(s, 0) for s in config.SFT_SPLITS)
    all_chars = sum(total_chars_by_split.values())
    chars_per_credit_oneway = credit / rate * 1e6  # chars buyable one-way

    return {
        "assumptions": {
            "google_translate_usd_per_million_chars": rate,
            "gcp_free_trial_credit_usd": credit,
            "sinhala_expansion_factor": exp,
            "note": ("List price; verify current Google Cloud Translation "
                     "pricing. Round-trip = forward(en) + back(si) legs."),
        },
        "per_split": {s: block(c) for s, c in total_chars_by_split.items()},
        "sft_only": block(sft_chars),
        "whole_dataset": block(all_chars),
        "chars_buyable_with_credit_one_way": int(chars_per_credit_oneway),
    }


def main() -> int:
    splits = require_splits(log)
    per_split = {}
    for split in config.ordered_splits():
        with StepTimer(log, f"text_stats[{split}]"):
            per_split[split] = analyze_split(split, splits[split])
            log.info("  %s: %s chars, ~%s tokens (%s)",
                     split, fmt_int(per_split[split]["total_chars"]),
                     fmt_int(per_split[split]["approx_total_tokens"]),
                     TOKEN_METHOD)

    total_chars_by_split = {s: per_split[s]["total_chars"] for s in per_split}
    overall = {
        "total_chars": sum(total_chars_by_split.values()),
        "approx_total_tokens": sum(s["approx_total_tokens"] for s in per_split.values()),
        "token_method": TOKEN_METHOD,
    }
    save_result(OUTPUT, {
        "per_split": per_split,
        "overall": overall,
        "cost_model": cost_model(total_chars_by_split),
    })
    log.info("Wrote %s (total %s chars)", OUTPUT, fmt_int(overall["total_chars"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
