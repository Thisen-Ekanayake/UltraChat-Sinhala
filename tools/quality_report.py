#!/usr/bin/env python3
"""Quality report for translated UltraChat-Sinhala JSONL shards.

Streams each split's shards once and computes structural, linguistic, and
orthographic quality metrics on the translated (Sinhala) output, then writes a
single Markdown report. Stdlib-only and deterministic, so it is a reproducible
dataset-prep artifact for the paper.

Record schema (per line): {"prompt": str, "prompt_id": str,
                           "messages": [{"role": "user"|"assistant", "content": str}, ...]}

Metrics (all computed over the *translated* text; no source alignment needed):
  * Structural integrity  — JSON validity, schema presence, user/assistant
    alternation, empty turns.
  * Language / script     — share of Sinhala vs Latin letters per message;
    "low-Sinhala" (<50% Sinhala letters) and "untranslated" (0 Sinhala, >=20
    Latin letters) message rates flag content NLLB left in English.
  * Failure modes         — empty outputs and degenerate repetition (detected by
    zlib compression ratio on messages >=200 chars; very compressible == looping).
  * Orthography (ZWJ)     — U+200D joiners present (the cleaning's positive
    signal) and remaining virama+space+consonant sequences (mostly *legitimate*
    word-final virama; see the repair audit for the lexicon-classified residual).
  * Duplication           — repeated prompt_ids and repeated content (md5 of the
    concatenated turns).
  * Size distribution     — per-message and per-record character lengths, turns.

Usage:
  python3 tools/quality_report.py --out results/quality_report.md \
      "SFT (cleaned)=data/translated_part_*/part_*.sinhala.jsonl" \
      "GEN (cleaned)=data/translated_gen_*/gen_*.sinhala.jsonl"
"""
import argparse
import glob
import hashlib
import json
import re
import zlib
from collections import defaultdict

SINH_LETTER = re.compile(r"[ක-ෆ]")          # Sinhala consonants/letters
LATIN = re.compile(r"[A-Za-z]")
RESID = re.compile(r"්[  ]+[ක-ෆ]")  # virama + space(s) + consonant
ZWJ = "‍"

LOW_LETTERS = 20      # min letters before judging a message's language
LOW_SINH_RATIO = 0.5  # below this Sinhala share -> "low-Sinhala"
REP_MINLEN = 200      # min chars before checking repetition
REP_RATIO = 0.18      # zlib ratio below this -> degenerate repetition
EX_MAX = 6            # examples kept per flag category


def q(sorted_vals, frac):
    if not sorted_vals:
        return 0
    n = len(sorted_vals)
    return sorted_vals[min(n - 1, int(frac * n))]


def mean(vals):
    return (sum(vals) / len(vals)) if vals else 0.0


def analyze(label, files):
    s = {
        "label": label, "files": files,
        "records": 0, "json_bad": 0, "schema_bad": 0,
        "messages": 0, "empty_msgs": 0, "role_alt_viol": 0,
        "sinh_ratios": [], "low_sinhala": 0, "untranslated": 0,
        "rep_ratios": [], "rep_flag": 0,
        "zwj_total": 0, "recs_with_zwj": 0, "resid_total": 0,
        "dup_pid": 0, "dup_content": 0,
        "msg_chars": [], "rec_chars": [], "turns": [],
        "ex": defaultdict(list),
    }
    seen_pid, seen_content = set(), set()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                s["records"] += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    s["json_bad"] += 1
                    continue
                msgs = rec.get("messages")
                pid = rec.get("prompt_id")
                if "prompt" not in rec or not isinstance(msgs, list) or not msgs:
                    s["schema_bad"] += 1
                    continue
                if pid in seen_pid:
                    s["dup_pid"] += 1
                else:
                    seen_pid.add(pid)

                rec_char, turns, prev_role, has_zwj, concat = 0, 0, None, False, []
                for m in msgs:
                    role, c = m.get("role"), m.get("content", "")
                    s["messages"] += 1
                    if role in ("user", "assistant"):
                        turns += 1
                    if prev_role is not None and role == prev_role:
                        s["role_alt_viol"] += 1
                    prev_role = role
                    cs = c.strip()
                    if not cs:
                        s["empty_msgs"] += 1
                        if len(s["ex"]["empty"]) < EX_MAX:
                            s["ex"]["empty"].append((pid, "<empty turn>"))
                        continue
                    L = len(c)
                    rec_char += L
                    s["msg_chars"].append(L)
                    concat.append(c)
                    sinh = len(SINH_LETTER.findall(c))
                    lat = len(LATIN.findall(c))
                    letters = sinh + lat
                    if letters >= LOW_LETTERS:
                        ratio = sinh / letters
                        s["sinh_ratios"].append(ratio)
                        if ratio < LOW_SINH_RATIO:
                            s["low_sinhala"] += 1
                            if len(s["ex"]["low_sinhala"]) < EX_MAX:
                                s["ex"]["low_sinhala"].append((pid, c[:200]))
                    if sinh == 0 and lat >= LOW_LETTERS:
                        s["untranslated"] += 1
                        if len(s["ex"]["untranslated"]) < EX_MAX:
                            s["ex"]["untranslated"].append((pid, c[:200]))
                    z = c.count(ZWJ)
                    if z:
                        s["zwj_total"] += z
                        has_zwj = True
                    s["resid_total"] += len(RESID.findall(c))
                    if L >= REP_MINLEN:
                        b = c.encode("utf-8")
                        r = len(zlib.compress(b, 6)) / len(b)
                        s["rep_ratios"].append(r)
                        if r < REP_RATIO:
                            s["rep_flag"] += 1
                            if len(s["ex"]["repetition"]) < EX_MAX:
                                s["ex"]["repetition"].append((pid, c[:200]))
                if has_zwj:
                    s["recs_with_zwj"] += 1
                s["rec_chars"].append(rec_char)
                s["turns"].append(turns)
                h = hashlib.md5("".join(concat).encode("utf-8")).hexdigest()
                if h in seen_content:
                    s["dup_content"] += 1
                else:
                    seen_content.add(h)
    return s


def pc(x, n):
    return f"{(100.0 * x / n):.3f}%" if n else "n/a"


def section(s):
    n = s["records"]
    nm = s["messages"]
    mc = sorted(s["msg_chars"])
    rc = sorted(s["rec_chars"])
    tu = sorted(s["turns"])
    sr = sorted(s["sinh_ratios"])
    rp = sorted(s["rep_ratios"])
    L = []
    L.append(f"## {s['label']}\n")
    L.append(f"- Shards analysed: **{len(s['files'])}**")
    L.append(f"- Records (dialogues): **{n:,}**  ·  messages (turns): **{nm:,}**\n")

    L.append("### Structural integrity\n")
    L.append("| check | count | rate |")
    L.append("|---|---:|---:|")
    L.append(f"| JSON-invalid lines | {s['json_bad']:,} | {pc(s['json_bad'], n)} |")
    L.append(f"| Schema-invalid records | {s['schema_bad']:,} | {pc(s['schema_bad'], n)} |")
    L.append(f"| Empty turns | {s['empty_msgs']:,} | {pc(s['empty_msgs'], nm)} |")
    L.append(f"| Role-alternation violations | {s['role_alt_viol']:,} | {pc(s['role_alt_viol'], nm)} |\n")

    L.append("### Language / script (translated text)\n")
    L.append("| metric | value |")
    L.append("|---|---:|")
    L.append(f"| Mean Sinhala-letter share per message | {mean(s['sinh_ratios']):.4f} |")
    L.append(f"| Median Sinhala-letter share | {q(sr, 0.5):.4f} |")
    L.append(f"| 1st-percentile Sinhala share | {q(sr, 0.01):.4f} |")
    L.append(f"| Low-Sinhala messages (<50% Sinhala letters) | {s['low_sinhala']:,} ({pc(s['low_sinhala'], nm)}) |")
    L.append(f"| Untranslated messages (0 Sinhala, ≥20 Latin) | {s['untranslated']:,} ({pc(s['untranslated'], nm)}) |\n")

    L.append("### Failure modes\n")
    L.append("| metric | value |")
    L.append("|---|---:|")
    L.append(f"| Empty turns | {s['empty_msgs']:,} ({pc(s['empty_msgs'], nm)}) |")
    L.append(f"| Degenerate-repetition messages (zlib ratio < {REP_RATIO}, ≥{REP_MINLEN} chars) | {s['rep_flag']:,} ({pc(s['rep_flag'], nm)}) |")
    L.append(f"| Median compression ratio (long messages) | {q(rp, 0.5):.3f} |\n")

    L.append("### Orthography — ZWJ / conjuncts\n")
    L.append("| metric | value |")
    L.append("|---|---:|")
    L.append(f"| Total ZWJ (U+200D) joiners | {s['zwj_total']:,} |")
    L.append(f"| Records containing ≥1 joiner | {s['recs_with_zwj']:,} ({pc(s['recs_with_zwj'], n)}) |")
    L.append(f"| Mean joiners per record | {s['zwj_total'] / n:.1f} |")
    L.append(f"| Virama+space+consonant remaining¹ | {s['resid_total']:,} |\n")
    L.append("> ¹ Mostly *legitimate* word-final virama before the next word, not errors. "
             "The lexicon-gated repair already merged the ~1.3M attested conjuncts per shard; "
             "see the repair audit (`sft_zwj_audit.md`) for the small out-of-scope residual.\n")

    L.append("### Duplication\n")
    L.append("| check | count | rate |")
    L.append("|---|---:|---:|")
    L.append(f"| Duplicate prompt_ids | {s['dup_pid']:,} | {pc(s['dup_pid'], n)} |")
    L.append(f"| Duplicate content (md5 of turns) | {s['dup_content']:,} | {pc(s['dup_content'], n)} |\n")

    L.append("### Size distribution\n")
    L.append("| metric | p50 | p90 | p99 | max |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(f"| Message chars | {q(mc, .5):,} | {q(mc, .9):,} | {q(mc, .99):,} | {(mc[-1] if mc else 0):,} |")
    L.append(f"| Record chars | {q(rc, .5):,} | {q(rc, .9):,} | {q(rc, .99):,} | {(rc[-1] if rc else 0):,} |")
    L.append(f"| Turns / record | {q(tu, .5):,} | {q(tu, .9):,} | {q(tu, .99):,} | {(tu[-1] if tu else 0):,} |\n")
    return "\n".join(L)


def examples(s):
    titles = {
        "untranslated": "Untranslated (Latin-only) messages",
        "low_sinhala": "Low-Sinhala (<50%) messages",
        "repetition": "Degenerate-repetition messages",
        "empty": "Empty turns",
    }
    out = [f"### {s['label']}\n"]
    any_ex = False
    for cat, title in titles.items():
        ex = s["ex"].get(cat) or []
        if not ex:
            continue
        any_ex = True
        out.append(f"**{title}** (showing {len(ex)}):\n")
        for pid, snip in ex:
            snip = snip.replace("\n", " ").replace("|", "\\|")
            out.append(f"- `{(pid or '?')[:12]}…` — {snip}")
        out.append("")
    if not any_ex:
        out.append("_No flagged examples in any category._\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("splits", nargs="+", help='LABEL=glob entries')
    args = ap.parse_args()

    stats = []
    for entry in args.splits:
        label, _, pat = entry.partition("=")
        files = sorted(glob.glob(pat))
        if not files:
            raise SystemExit(f"no files match: {pat}")
        print(f"analysing {label}: {len(files)} files ...", flush=True)
        stats.append(analyze(label, files))

    import datetime
    md = []
    md.append("# UltraChat-Sinhala — Translation Quality Report\n")
    md.append(f"_Generated {datetime.datetime.now():%Y-%m-%d %H:%M} · "
              f"`tools/quality_report.py` (full-scan, deterministic)_\n")
    md.append("This report measures the **translated Sinhala output** of the cleaned dataset "
              "(post ZWJ repair) across structural, linguistic, orthographic, and duplication "
              "dimensions. All metrics are computed over every record (no sampling).\n")

    # headline comparison
    md.append("## Summary\n")
    md.append("| split | records | turns | JSON-invalid | untranslated turns | repetition turns | dup prompt_ids | recs w/ ZWJ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        n, nm = s["records"], s["messages"]
        md.append(f"| {s['label']} | {n:,} | {nm:,} | {pc(s['json_bad'], n)} | "
                  f"{pc(s['untranslated'], nm)} | {pc(s['rep_flag'], nm)} | "
                  f"{pc(s['dup_pid'], n)} | {pc(s['recs_with_zwj'], n)} |")
    md.append("")

    md.append("## Methodology\n")
    md.append(
        "- **Scope:** full scan of all shards; metrics computed on the translated text only "
        "(source-aligned length-ratio is noted as future work).\n"
        "- **Language/script:** per message, the Sinhala-letter share = "
        "`Sinhala letters / (Sinhala + Latin letters)`, judged only when ≥20 letters are present. "
        "A *low-Sinhala* message scores <0.5; an *untranslated* message has 0 Sinhala letters and "
        "≥20 Latin (NLLB left it in English). Some Latin is expected and legitimate — code, URLs, "
        "identifiers, brand names.\n"
        f"- **Repetition:** for messages ≥{REP_MINLEN} chars, the zlib compression ratio "
        "(compressed/raw bytes) is a degeneracy proxy; ratios below "
        f"{REP_RATIO} indicate looping/repeated spans.\n"
        "- **Orthography:** ZWJ (U+200D) counts confirm the conjunct repair landed; "
        "virama+space+consonant counts are reported for transparency but are dominated by "
        "legitimate word boundaries.\n"
        "- **Duplication:** exact duplicate `prompt_id`s and exact duplicate concatenated content (md5).\n")

    for s in stats:
        md.append(section(s))

    md.append("## Flagged examples (qualitative)\n")
    for s in stats:
        md.append(examples(s))

    md.append("## Limitations\n")
    md.append(
        "- No reference translations, so this measures **dataset hygiene and fluency proxies**, "
        "not adequacy/accuracy (no BLEU/COMET vs a gold set).\n"
        "- Source-aligned target/source length ratios (truncation/expansion detection) are not yet "
        "included; they require joining to the English parquet by `prompt_id`.\n"
        "- The repetition and language thresholds are heuristics; the per-category example lists "
        "above are provided so the thresholds can be eyeballed and tuned.\n")

    text = "\n".join(md) + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\nwrote {args.out} ({len(text):,} bytes)")
    # also echo the summary table to stdout
    for s in stats:
        print(f"  {s['label']}: {s['records']:,} recs, "
              f"untranslated={pc(s['untranslated'], s['messages'])}, "
              f"repetition={pc(s['rep_flag'], s['messages'])}, "
              f"dup_pid={s['dup_pid']:,}, zwj_recs={pc(s['recs_with_zwj'], s['records'])}")


if __name__ == "__main__":
    main()
