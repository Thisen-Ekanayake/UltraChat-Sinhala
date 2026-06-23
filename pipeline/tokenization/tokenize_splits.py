#!/usr/bin/env python3
"""Stage 7 — tokenise the corpus with the SinLlama tokenizer.

Tokenises every message with the SinLlama merged model's tokenizer (a
Sinhala-extended Llama tokenizer) and reports token statistics **per split**
and **grouped by `sft` vs `gen` separately**, since the two families serve
different downstream purposes.

For each scope it reports:
  * total tokens and mean tokens-per-character (a proxy for tokenizer fertility
    on this text);
  * message-level and dialogue-level token-length distributions (percentiles);
  * dialogues whose assembled token length exceeds common context windows
    (2048/4096/8192) — i.e. SFT samples that would be truncated.

Token counts use raw content tokens (``add_special_tokens=False``); the
per-dialogue total adds a small fixed overhead per turn to approximate chat
templating (see TURN_OVERHEAD_TOKENS), which is reported alongside the raw sum.

This stage is standalone (not wired into scripts/run_analysis.sh) because it
requires the SinLlama model directory, which is provided separately on the VM.
Point it at the model with ``UC_TOKENIZER_DIR`` (see config.py).

Output: results/07_tokenization.json

Usage (run from the repo root):
    python -m pipeline.tokenization.tokenize_splits                 # all discovered splits
    python -m pipeline.tokenization.tokenize_splits --splits train_sft test_sft
    UC_TOKENIZER_DIR=/models/SinLlama_merged_bf16 python -m pipeline.tokenization.tokenize_splits
"""
from __future__ import annotations

import argparse
import sys

from pipeline import config
from pipeline.common import (
    LengthAccumulator, StepTimer, fmt_int, get_logger,
    iter_messages, iter_split_records, require_splits, save_result,
)

log = get_logger("tokenize")
OUTPUT = "07_tokenization.json"

# Approximate per-turn overhead for chat templating (role markers / separators),
# used only for the context-window overflow estimate.
TURN_OVERHEAD_TOKENS = 4


# ---------------------------------------------------------------------------
# Tokeniser loading (transformers fast tokenizer, fallback to `tokenizers`)
# ---------------------------------------------------------------------------
def load_tokenizer():
    tdir = config.TOKENIZER_DIR
    if not tdir.exists():
        log.error("Tokenizer dir not found: %s\n"
                  "Set UC_TOKENIZER_DIR to the SinLlama model directory.", tdir)
        sys.exit(2)

    # Preferred: HuggingFace fast tokenizer (honours tokenizer_config.json).
    # local_files_only=True forbids any Hub/network lookup — the model is always
    # local here, and this avoids the confusing "Repo id must be in the form …"
    # error transformers raises when it mistakes a local path for a repo id.
    try:
        from transformers import AutoTokenizer
    except ImportError:
        AutoTokenizer = None

    if AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(
                str(tdir), use_fast=True, local_files_only=True)

            def encode_batch(texts: list[str]) -> list[int]:
                enc = tok(texts, add_special_tokens=False)["input_ids"]
                return [len(ids) for ids in enc]

            # len(tok) includes added tokens (the Sinhala extension);
            # tok.vocab_size reports only the base vocab and would undercount.
            full_vocab = len(tok)
            meta = {
                "backend": "transformers.AutoTokenizer",
                "vocab_size": full_vocab,
                "base_vocab_size": tok.vocab_size,
                "tokenizer_dir": str(tdir),
            }
            log.info("Loaded transformers tokenizer (vocab=%s, base=%s) from %s",
                     fmt_int(full_vocab), fmt_int(tok.vocab_size), tdir)
            return encode_batch, meta
        except Exception as exc:
            log.warning("transformers load failed (%s); trying `tokenizers`.", exc)

    # Fallback: lightweight Rust tokenizer straight from tokenizer.json. This is
    # sufficient for counting and needs no transformers/torch install.
    try:
        from tokenizers import Tokenizer
    except ImportError:
        log.error(
            "No tokenizer backend available (neither `transformers` nor "
            "`tokenizers` is installed).\n"
            "Install the lightweight backend (enough for token counting):\n"
            "    pip install tokenizers\n"
            "On an externally-managed VM (PEP 668), use one of:\n"
            "    pip install --user tokenizers\n"
            "    pip install --break-system-packages tokenizers"
        )
        sys.exit(3)

    tjson = tdir / "tokenizer.json"
    if not tjson.is_file():
        log.error("tokenizer.json not found in %s", tdir)
        sys.exit(3)
    tok = Tokenizer.from_file(str(tjson))

    def encode_batch(texts: list[str]) -> list[int]:
        encs = tok.encode_batch(texts, add_special_tokens=False)
        return [len(e.ids) for e in encs]

    meta = {
        "backend": "tokenizers.Tokenizer",
        "vocab_size": tok.get_vocab_size(),
        "tokenizer_dir": str(tdir),
    }
    log.info("Loaded `tokenizers` backend (vocab=%s) from %s",
             fmt_int(meta["vocab_size"]), tdir)
    return encode_batch, meta


# ---------------------------------------------------------------------------
# Per-split tokenisation pass
# ---------------------------------------------------------------------------
# Each dialogue's messages are encoded as one batch: dialogues are small, so
# this is efficient while keeping the message<->dialogue mapping exact (needed
# for the per-dialogue context-window overflow counts).
def tokenize_split(split: str, shards, encode_batch) -> dict:
    msg_tok = LengthAccumulator()
    dlg_tok_raw = LengthAccumulator()
    dlg_tok_chat = LengthAccumulator()
    total_tokens = 0
    total_chars = 0
    over_ctx = {w: 0 for w in config.CONTEXT_WINDOWS}

    for rec in iter_split_records(split, shards):
        contents = [m["content"] for m in iter_messages(rec)]
        if not contents:
            dlg_tok_raw.add(0)
            dlg_tok_chat.add(0)
            continue
        # encode this dialogue's messages (batched per dialogue; dialogues are
        # small, so this is efficient and keeps message<->dialogue mapping exact)
        lengths = encode_batch(contents)
        for n in lengths:
            msg_tok.add(n)
        d_raw = sum(lengths)
        d_chat = d_raw + TURN_OVERHEAD_TOKENS * len(lengths)
        total_tokens += d_raw
        total_chars += sum(len(c) for c in contents)
        dlg_tok_raw.add(d_raw)
        dlg_tok_chat.add(d_chat)
        for w in config.CONTEXT_WINDOWS:
            if d_chat > w:
                over_ctx[w] += 1

    return {
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "tokens_per_char": round(total_tokens / total_chars, 4) if total_chars else 0.0,
        "message_tokens": msg_tok.summary(),
        "dialogue_tokens_raw": dlg_tok_raw.summary(),
        "dialogue_tokens_chat_est": dlg_tok_chat.summary(),
        "dialogues_over_context_window": {str(w): over_ctx[w] for w in config.CONTEXT_WINDOWS},
        "turn_overhead_tokens": TURN_OVERHEAD_TOKENS,
    }


def aggregate(groups: dict[str, list[str]], per_split: dict) -> dict:
    """Aggregate scalar totals + overflow counts for a group of splits."""
    out = {}
    for group, members in groups.items():
        members = [s for s in members if s in per_split]
        if not members:
            continue
        tot_tokens = sum(per_split[s]["total_tokens"] for s in members)
        tot_chars = sum(per_split[s]["total_chars"] for s in members)
        over = {str(w): 0 for w in config.CONTEXT_WINDOWS}
        for s in members:
            for w, v in per_split[s]["dialogues_over_context_window"].items():
                over[w] += v
        out[group] = {
            "splits": members,
            "total_tokens": tot_tokens,
            "total_chars": tot_chars,
            "tokens_per_char": round(tot_tokens / tot_chars, 4) if tot_chars else 0.0,
            "dialogues_over_context_window": over,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Restrict to these splits (default: all discovered).")
    args = ap.parse_args()

    splits = require_splits(log)
    encode_batch, meta = load_tokenizer()

    selected = args.splits or config.ordered_splits()
    per_split = {}
    for split in selected:
        if split not in splits:
            log.warning("Split %s not found; skipping.", split)
            continue
        with StepTimer(log, f"tokenize[{split}]"):
            per_split[split] = tokenize_split(split, splits[split], encode_batch)
            s = per_split[split]
            log.info("  %s: %s tokens, %.3f tok/char, median msg %.0f tok",
                     split, fmt_int(s["total_tokens"]), s["tokens_per_char"],
                     s["message_tokens"]["p50"])

    grouped = aggregate(config.SPLIT_GROUPS, per_split)
    overall = {
        "total_tokens": sum(s["total_tokens"] for s in per_split.values()),
        "total_chars": sum(s["total_chars"] for s in per_split.values()),
    }
    if overall["total_chars"]:
        overall["tokens_per_char"] = round(
            overall["total_tokens"] / overall["total_chars"], 4)

    save_result(OUTPUT, {
        "tokenizer": meta,
        "per_split": per_split,
        "by_group": grouped,
        "overall": overall,
    })
    log.info("Wrote %s", OUTPUT)
    for group, g in grouped.items():
        log.info("  GROUP %s: %s tokens across %s",
                 group.upper(), fmt_int(g["total_tokens"]), g["splits"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
