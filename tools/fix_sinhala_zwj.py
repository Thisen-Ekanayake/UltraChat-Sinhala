#!/usr/bin/env python3
"""Batch ZWJ-repair tool for already-translated Sinhala JSONL files.

NLLB strips the Zero-Width Joiner (U+200D) from Sinhala conjuncts — its
SentencePiece normaliser maps ``U+0DCA U+200D <C>`` -> ``U+0DCA U+0020 <C>`` at
encode time — so existing translation output has ``ප් ර`` where ``ප්‍ර`` belongs.
This tool restores the joiner across whole JSONL shards.

It is a thin CLI over the canonical, test-validated normaliser core in
``pipeline/translation/sinhala_normalize.py`` (the same algorithm the translation
stage now applies live, so re-translated data needs no repair). Method, tiers and
guarantees are documented there and in ``docs/sinhala_zwj_repair.md``;
``tools/test_fix_sinhala_zwj.py`` machine-checks them. Only a single ``U+0020``
is ever swapped for ``U+200D``, so JSONL structure is preserved (``--apply`` also
aborts on any invalid JSON).

Examples
--------
  # dry-run report (builds/caches the lexicon from the corpus on first use):
  python3 tools/fix_sinhala_zwj.py data/translated/test_sft.sinhala.jsonl \\
      --lexicon-cache data/sinhala_lexicon.pkl --report results/audit.md
  # apply in place (keeps a .bak):
  python3 tools/fix_sinhala_zwj.py FILE.jsonl --lexicon-cache ... --apply
"""
import argparse
import io
import json
import os
import sys

# Import the canonical core from the pipeline package (repo root on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.translation.sinhala_normalize import (   # noqa: E402
    ZWJ, build_lexicon, is_lone_hal, load_lexicon, load_vocab_zwj,
    new_stats, repair_text, valid_conjunct_clusters,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="JSONL file(s) to repair")
    ap.add_argument("--corpus", default="CPT-Dataset.txt")
    ap.add_argument("--vocab", default="tokenizer/unigram_32000_0.9995.vocab")
    ap.add_argument("--lexicon-cache", default=None, help="pickle cache for the lexicon")
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--apply", action="store_true",
                    help="overwrite input in place (a .bak backup is kept)")
    ap.add_argument("--report", default=None, help="write a detailed audit report here")
    args = ap.parse_args()

    W = load_lexicon(args.corpus, args.lexicon_cache)
    vocab_zwj = load_vocab_zwj(args.vocab)
    valid_clusters = valid_conjunct_clusters(W)
    sys.stderr.write(f"lexicon: {len(W):,} types | vocab ZWJ pieces: {len(vocab_zwj):,} | "
                     f"attested ර/ය clusters: {len(valid_clusters)}\n")

    report = io.open(args.report, "w", encoding="utf-8") if args.report else None
    for path in args.inputs:
        text = io.open(path, encoding="utf-8").read()
        st = new_stats()
        new_text = repair_text(text, W, vocab_zwj, args.threshold, st, valid_clusters)

        bad = sum(1 for ln in new_text.splitlines()
                  if ln.strip() and _json_bad(ln))
        before, after = text.count(ZWJ), new_text.count(ZWJ)
        print(f"\n{path}")
        print(f"  merge sites          : {st['merge_sites']:,}  "
              f"(T1={st['tier']['T1']:,} T2={st['tier']['T2']:,} T3={st['tier']['T3']:,})")
        print(f"  distinct merged forms: {len(st['merged_forms']):,}")
        print(f"  kept hal+space sites : {st['kept_sites']:,}")
        print(f"  residual lone-hal    : {sum(st['residual'].values()):,} "
              f"({len(st['residual'])} distinct)")
        print(f"  ZWJ U+200D           : {before:,} -> {after:,}")
        print(f"  JSON validity        : {'ALL VALID' if bad == 0 else f'{bad} BAD LINES'}")

        if report:
            _write_report(report, path, st)
        if args.apply:
            if bad:
                sys.exit(f"ABORT: {bad} invalid JSON lines in {path}; not writing.")
            os.replace(path, path + ".bak")
            io.open(path, "w", encoding="utf-8").write(new_text)
            print(f"  applied (backup: {path}.bak)")
    if report:
        report.close()


def _json_bad(line):
    try:
        json.loads(line); return False
    except Exception:
        return True


def _write_report(report, path, st):
    report.write(f"# Audit report: {path}\n\n")
    report.write("## Tier-3 (structural, lowest confidence — please eyeball)\n")
    for cand, c in st["t3_audit"].most_common():
        report.write(f"  {cand}   x{c}\n")
    report.write("\n## Residual lone-hal NOT merged (out-of-scope / non-conjunct artifacts)\n")
    for (l, r), c in st["residual"].most_common():
        report.write(f"  {l} | {r}   x{c}\n")
    report.write("\n## All distinct merges (form, corpus-freq, tier)\n")
    for cand, (fr, tier) in sorted(st["merged_forms"].items(), key=lambda x: -x[1][0]):
        report.write(f"  {cand}\t{fr}\t{tier}\n")


if __name__ == "__main__":
    main()
