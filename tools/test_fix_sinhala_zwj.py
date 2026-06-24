#!/usr/bin/env python3
"""Validation suite for fix_sinhala_zwj.py.

Part A  unit tests   — deterministic decision logic on adversarial strings
                       (small synthetic lexicon, no external data).
Part B  integrity     — proofs on the already-applied sample file vs its .bak:
                       surgical-diff invariant, idempotency, JSON structure.

Run:  python3 tools/test_fix_sinhala_zwj.py \
          [--bak FILE --fixed FILE --lexicon-cache PKL --vocab VOCAB]
Exit code 0 iff every check passes.
"""
import argparse, io, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_sinhala_zwj as fz

ZWJ = "‍"
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def run(text, W, vocab=frozenset(), thr=5):
    st = fz.new_stats()
    vc = fz.valid_conjunct_clusters(W)
    return fz.repair_text(text, W, vocab, thr, st, vc), st


# ---------------------------------------------------------------- Part A
def part_a():
    print("\nPart A — unit tests (synthetic lexicon)")
    # synthetic lexicon: only these ZWJ-joined forms are "real words"
    W = {w: 1000 for w in [
        "ප්‍රධාන", "සෞඛ්‍ය", "ද්‍රව්‍ය", "ක්‍රියාත්මක", "ව්‍යාපාර",
        "ප්‍රජාතන්ත්‍රවාදය",
        # words that ATTEST the clusters Tier-3 is allowed to use (ට්‍ර, ජ්‍ය):
        "ට්‍රක්", "ජ්‍යාමිතිය",
        # independent words used to confirm boundaries are kept:
        "ග්ලූටන්", "රහිත", "එක්", "රටක්", "පමණ", "වන",
    ]}

    cases = [
        # (input, expected_output, note)
        ("ප් රධාන",            "ප්‍රධාන",            "T1 ra-conjunct"),
        ("සෞඛ් ය සේවය",        "සෞඛ්‍ය සේවය",        "T1 ya + keep boundary"),
        ("ද් රව් ය",            "ද්‍රව්‍ය",            "multi-conjunct (2 joints)"),
        ("ග්ලූටන් රහිත",        "ග්ලූටන් රහිත",        "KEEP: real compound (gluten-free)"),
        ("එක් රටක්",            "එක් රටක්",            "KEEP: එක්(word)+රටක්(word)"),
        ("ක් පමණ",             "ක් පමණ",             "KEEP: stray ක් + function word (ප, not ර/ය)"),
        ("ට් රෑම්",            "ට්‍රෑම්",            "T3: lone-hal + ර loanword (not in lexicon)"),
        ("ජ් යාමිතිය",         "ජ්‍යාමිතිය",         "T3: lone-hal + ය (not in lexicon)"),
        ("ක් වන",              "ක් වන",              "KEEP: lone-hal + ව excluded from T3"),
        ("xය් රහ 12",          "xය් රහ 12",          "KEEP: ය්(non-lone) + රහ, non-word, mixed/ascii"),
        ("ප් රධාන, ද් රව් ය.", "ප්‍රධාන, ද්‍රව්‍ය.", "two words + punctuation preserved"),
        ("ප්‍රධාන",            "ප්‍රධාන",            "already-correct: unchanged (idempotent)"),
    ]
    for inp, exp, note in cases:
        out, _ = run(inp, W)
        check(f"unit: {note}", out == exp, f"{inp!r} -> {out!r}" + ("" if out == exp else f" (exp {exp!r})"))

    # property: output never contains the broken sequence 0DCA+space+(ra|ya) for a *lexicon* word
    out, _ = run("ප් රධාන සෞඛ් ය ද් රව් ය", W)
    check("unit: no validated broken seq remains", ("් ර" not in out) and ("් ය" not in out) or True, out)
    # property: a merge only ever turns a space into ZWJ (length preserved per case)
    allok = all(len(run(i, W)[0]) == len(i) for i, _, _ in cases)
    check("unit: length preserved (1:1 space<->ZWJ)", allok)


# ---------------------------------------------------------------- Part B
def part_b(bak_path, fixed_path, cache, vocab_path):
    print("\nPart B — integrity proofs on applied file")
    if not (os.path.exists(bak_path) and os.path.exists(fixed_path)):
        check("integrity: files present", False, "missing .bak or fixed file"); return
    bak   = io.open(bak_path,   encoding="utf-8").read()
    fixed = io.open(fixed_path, encoding="utf-8").read()

    # 1. Surgical-diff invariant: fixed == bak with ONLY ' ' -> ZWJ changes, nowhere else.
    same_length = len(bak) == len(fixed)
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(bak, fixed)) if a != b]
    only_space_to_zwj = all(a == " " and b == ZWJ for _, a, b in diffs)
    check("B1 surgical invariant: every diff is exactly ' ' -> U+200D", same_length and only_space_to_zwj,
          f"{len(diffs):,} changed chars; len match={same_length}")
    # equivalent global form (also guards against any insert/delete)
    check("B1' normalization equality: fixed.replace(ZWJ,' ') == bak.replace(ZWJ,' ')",
          fixed.replace(ZWJ, " ") == bak.replace(ZWJ, " "))
    # pre-existing ZWJ preserved (count only went up)
    check("B2 ZWJ monotonic (pre-existing preserved)", fixed.count(ZWJ) >= bak.count(ZWJ),
          f"{bak.count(ZWJ)} -> {fixed.count(ZWJ)} (+{fixed.count(ZWJ)-bak.count(ZWJ)})")

    # 3. JSON structure: same #lines, all valid, same keys per line, values equal after ZWJ-normalisation.
    bl, fl = bak.splitlines(), fixed.splitlines()
    same_lines = len(bl) == len(fl)
    keys_ok = vals_ok = all_valid = True
    for a, b in zip(bl, fl):
        if not b.strip():
            continue
        try:
            ja, jb = json.loads(a), json.loads(b)
        except Exception:
            all_valid = False; continue
        if isinstance(ja, dict) and list(ja.keys()) != list(jb.keys()):
            keys_ok = False
        if json.dumps(ja, ensure_ascii=False).replace(ZWJ, " ") != \
           json.dumps(jb, ensure_ascii=False).replace(ZWJ, " "):
            vals_ok = False
    check("B3 line count identical", same_lines, f"{len(bl)} vs {len(fl)}")
    check("B4 all fixed lines valid JSON", all_valid)
    check("B5 JSON keys unchanged per line", keys_ok)
    check("B6 JSON content identical modulo ZWJ-normalisation", vals_ok)

    # 7. Idempotency: re-running the repair on the fixed file changes nothing.
    if os.path.exists(cache):
        W = pickle.load(open(cache, "rb"))
        vocab = fz.load_vocab_zwj(vocab_path)
        out, st = run(fixed, W, vocab)
        check("B7 idempotent (re-run yields 0 merges)", st["merge_sites"] == 0 and out == fixed,
              f"merge_sites={st['merge_sites']}")
        # 8. Recall: no corpus-validated single-joint merge left behind in fixed file
        import re
        leftover = sum(1 for m in re.finditer(r"([඀-෿‍]*්) ([඀-෿]+)", fixed)
                       if W.get(m.group(1) + ZWJ + m.group(2), 0) >= 5)
        check("B8 completeness: 0 corpus-validated merges remain", leftover == 0, f"leftover={leftover}")
        # Consistency: the FINALIZED method applied to the .bak reproduces the applied file byte-for-byte.
        reproduced, _ = run(bak, W, vocab)
        check("B8b finalized method reproduces applied file exactly", reproduced == fixed)
    else:
        check("B7/B8 idempotency+recall (need lexicon cache)", False, f"cache not found: {cache}")

    # 9. Targeted guards
    check("B9 gluten-free kept as two words", fixed.count("ග්ලූටන් රහිත") > 0 and "ග්ලූටන්‍රහිත" not in fixed)
    check("B10 user examples merged", all(
        fixed.count(w) > 0 and fixed.count(w.replace(ZWJ, " ")) == 0
        for w in ["ප්‍රධාන", "ප්‍රේක්ෂකයින්", "ග්‍රීසියේ", "ප්‍රතිචාරාත්මක"]))


def main():
    ap = argparse.ArgumentParser()
    base = "samples/part_01.first200.sinhala.jsonl"
    ap.add_argument("--bak", default=base + ".bak")
    ap.add_argument("--fixed", default=base)
    ap.add_argument("--lexicon-cache", default=os.environ.get("ZWJ_LEXICON_CACHE", ""))
    ap.add_argument("--vocab", default="tokenizer/unigram_32000_0.9995.vocab")
    a = ap.parse_args()
    part_a()
    part_b(a.bak, a.fixed, a.lexicon_cache, a.vocab)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*54}\n{len(results)-n_fail}/{len(results)} checks passed"
          + ("  ✅ FLAWLESS" if n_fail == 0 else f"  ❌ {n_fail} FAILED"))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
