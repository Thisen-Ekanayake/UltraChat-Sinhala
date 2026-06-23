#!/usr/bin/env python3
"""Stage 4 — Unicode / multilingual composition analysis.

UltraChat is nominally English, but ~10% of messages (observed on the test
split) carry non-ASCII content. Some is benign typography (smart quotes, en/em
dashes, the degree sign); some is genuinely foreign-script text (Devanagari,
Arabic, CJK, Cyrillic, ...). Translating *already-multilingual* source through
an en->si engine injects noise, so we quantify it here.

Per split and dataset-wide, computes:
  * messages containing any non-ASCII character (count + %)
  * total non-ASCII character volume
  * messages containing each Unicode script of interest (uses the `regex`
    module's \\p{Script=...} property when available)
  * messages already carrying Sinhala (the target script) before translation
  * separate tally of "typographic" non-ASCII (quotes/dashes/symbols) vs
    "foreign-script letters", so cosmetic noise is not mistaken for foreign text

Output: results/04_unicode_scripts.json
"""
from __future__ import annotations

import sys
from collections import Counter

import config
from uc_common import (
    StepTimer, fmt_int, get_logger, iter_messages, iter_split_records,
    require_splits, save_result,
)

log = get_logger("unicode")
OUTPUT = "04_unicode_scripts.json"

# Scripts whose presence signals genuinely foreign-language source text.
SCRIPTS = [
    "Sinhala", "Devanagari", "Tamil", "Bengali", "Arabic", "Hebrew",
    "Cyrillic", "Greek", "Han", "Hiragana", "Katakana", "Hangul",
    "Thai", "Gujarati", "Telugu", "Kannada", "Malayalam",
]

try:
    import regex as _re

    HAVE_REGEX = True
    _SCRIPT_PATS = {s: _re.compile(rf"\p{{Script={s}}}") for s in SCRIPTS}
    # "Foreign letters" = any letter that is not Latin/Common/Inherited.
    _FOREIGN_LETTER = _re.compile(r"[^\p{Script=Latin}\p{Script=Common}\p{Script=Inherited}]")
    # A letter specifically (to avoid counting symbols as foreign text).
    _IS_FOREIGN_TEXT = _re.compile(
        r"\p{L}", flags=0)  # placeholder; combined below
except Exception:  # pragma: no cover
    HAVE_REGEX = False
    _SCRIPT_PATS = {}


def classify_nonascii(text: str) -> tuple[int, int]:
    """Return (typographic_count, other_nonascii_count) for chars > 0x7F.

    "Typographic" = common punctuation/symbol code points (smart quotes,
    dashes, degree, bullet, currency, etc.) that are cosmetic for translation.
    """
    typo = 0
    other = 0
    for ch in text:
        o = ord(ch)
        if o < 128:
            continue
        cat_typo = (
            0x2010 <= o <= 0x2027  # dashes, quotes, bullets, ellipsis
            or 0x2030 <= o <= 0x205E  # per-mille, primes, misc punctuation
            or o in (0x00B0, 0x00A9, 0x00AE, 0x2122, 0x00A3, 0x20AC, 0x00A0,
                     0x00BD, 0x00BC, 0x00BE, 0x2212)
        )
        if cat_typo:
            typo += 1
        else:
            other += 1
    return typo, other


def analyze_split(split: str, shards) -> dict:
    n_messages = 0
    n_nonascii_msgs = 0
    nonascii_chars = 0
    typo_chars = 0
    foreign_chars = 0
    n_foreign_script_msgs = 0
    script_msg_counts = Counter()

    for rec in iter_split_records(split, shards):
        for m in iter_messages(rec):
            n_messages += 1
            text = m["content"]
            if text.isascii():
                continue
            n_nonascii_msgs += 1
            typo, other = classify_nonascii(text)
            nonascii_chars += typo + other
            typo_chars += typo
            foreign_chars += other
            if HAVE_REGEX:
                hit_foreign_script = False
                for s, pat in _SCRIPT_PATS.items():
                    if pat.search(text):
                        script_msg_counts[s] += 1
                        hit_foreign_script = True
                if hit_foreign_script:
                    n_foreign_script_msgs += 1

    return {
        "n_messages": n_messages,
        "n_nonascii_messages": n_nonascii_msgs,
        "pct_nonascii_messages": round(100 * n_nonascii_msgs / n_messages, 3) if n_messages else 0.0,
        "nonascii_chars": nonascii_chars,
        "typographic_chars": typo_chars,
        "foreign_script_chars": foreign_chars,
        "n_foreign_script_messages": n_foreign_script_msgs,
        "script_message_counts": dict(script_msg_counts),
        "regex_available": HAVE_REGEX,
    }


def main() -> int:
    splits = require_splits(log)
    if not HAVE_REGEX:
        log.warning("`regex` module unavailable; per-script breakdown skipped.")
    per_split = {}
    for split in config.ordered_splits():
        with StepTimer(log, f"unicode[{split}]"):
            per_split[split] = analyze_split(split, splits[split])
            log.info("  %s: %s non-ASCII msgs, %s foreign-script msgs",
                     split, fmt_int(per_split[split]["n_nonascii_messages"]),
                     fmt_int(per_split[split]["n_foreign_script_messages"]))

    n_messages = sum(s["n_messages"] for s in per_split.values())
    agg_scripts = Counter()
    for s in per_split.values():
        agg_scripts.update(s["script_message_counts"])
    overall = {
        "n_messages": n_messages,
        "n_nonascii_messages": sum(s["n_nonascii_messages"] for s in per_split.values()),
        "n_foreign_script_messages": sum(s["n_foreign_script_messages"] for s in per_split.values()),
        "nonascii_chars": sum(s["nonascii_chars"] for s in per_split.values()),
        "typographic_chars": sum(s["typographic_chars"] for s in per_split.values()),
        "foreign_script_chars": sum(s["foreign_script_chars"] for s in per_split.values()),
        "script_message_counts": dict(agg_scripts.most_common()),
    }
    save_result(OUTPUT, {"per_split": per_split, "overall": overall})
    log.info("Wrote %s", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
