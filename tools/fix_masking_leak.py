#!/usr/bin/env python3
"""Repair MT masking-sentinel leakage in translated UltraChat-Sinhala JSONL.

ROOT CAUSE
----------
The MT masking placeholder is ``⟦n⟧`` (``mt_preprocess.protect``), built from the
mathematical white brackets U+27E6/U+27E7. NLLB's SentencePiece normaliser strips
those rare brackets during translation, leaving a bare digit ``n``.
``mt_preprocess.restore`` matches placeholders with ``⟦\\s*(\\d+)\\s*⟧?`` — which
**requires the opening ⟦** — so it never matches the degraded bare-digit form.
Every masked span is therefore judged "missing" and **appended at the end of the
message**, while a stray digit is left inline where the span belonged. Measured
incidence: ~2.56% of messages carry masked spans, and ~100% of those leaked.

REPAIR (source-guided, line-aligned)
------------------------------------
For each translated message, joined to its English source by ``prompt_id``:
  1. ``protect(source)`` -> the exact original spans ``O`` and the masked text,
     which records *which placeholder index sits on which source line*.
  2. Strip the appended tail (``" ".join(O)``) that ``restore`` added at the end.
  3. ``from_segments`` preserves line count, so the body line-aligns with the
     masked source. On each line, replace the leaked bare digit for the
     placeholder index(es) known to belong there with the original span. Matching
     is by exact digit string and constrained to the indices expected on that
     line, so real numbers elsewhere (e.g. "divisible by 4") are never touched.
  4. Validate: every span reinserted exactly once. Any span that cannot be
     located (NLLB dropped/hallucinated the digit) is re-appended rather than
     lost; that message is counted "partial". Messages lacking the append
     signature are left byte-for-byte unchanged ("skipped").

Usage:
  python3 tools/fix_masking_leak.py --src SRC.parquet --in IN.jsonl --report
  python3 tools/fix_masking_leak.py --src SRC.parquet --in IN.jsonl --out OUT.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.translation.mt_preprocess import protect  # noqa: E402

import pandas as pd  # noqa: E402

_PH_IN_MASKED = re.compile(r"⟦\s*(\d+)\s*⟧")
_STANDALONE = re.compile(r"(?<![\d⟦])(\d{1,3})(?![\d⟧])")


def _reinsert(body: str, masked: str, O: list) -> tuple[str, list]:
    """Line-align body with masked source; replace leaked digits in place.

    Returns (new_body, used) where used[i] is True if span i was reinserted.
    """
    used = [False] * len(O)
    blines = body.split("\n")
    mlines = masked.split("\n")
    if len(blines) != len(mlines):
        return body, used  # structure diverged -> signal caller to fall back

    out_lines = []
    for bline, mline in zip(blines, mlines):
        want = [int(x) for x in _PH_IN_MASKED.findall(mline)]
        if not want:
            out_lines.append(bline)
            continue
        res, last, wi = [], 0, 0
        for m in _STANDALONE.finditer(bline):
            if wi >= len(want):
                break
            if m.group(1) == str(want[wi]):
                res.append(bline[last:m.start()])
                res.append(O[want[wi]])
                last = m.end()
                used[want[wi]] = True
                wi += 1
        res.append(bline[last:])
        out_lines.append("".join(res))
    return "\n".join(out_lines), used


def fix_text(T: str, src: str) -> tuple[str, str]:
    """Repair one translated string given its English source. Returns (text, status)."""
    masked, O = protect(src)
    if not O:
        return T, "no_mask"
    tail = " ".join(O)
    Ts = T.rstrip()
    if not tail or not Ts.endswith(tail):
        return T, "skipped_no_tail"
    body = Ts[: len(Ts) - len(tail)].rstrip()
    new_body, used = _reinsert(body, masked, O)
    missing = [O[i] for i in range(len(O)) if not used[i]]
    if not missing:
        return new_body, "reconstructed"
    # safe fallback: never drop content — re-append whatever we couldn't place.
    return new_body.rstrip() + "\n\n" + "\n\n".join(missing), "partial"


def fix_record(rec: dict, srec: dict, st: Counter) -> bool:
    """Repair prompt + messages in-place. Returns True if anything changed."""
    changed = False
    smsgs = [m.get("content", "") for m in srec.get("messages", [])]
    for i, m in enumerate(rec.get("messages", [])):
        if i >= len(smsgs):
            continue
        new, status = fix_text(m["content"], smsgs[i])
        st[status] += 1
        if status in ("reconstructed", "partial") and new != m["content"]:
            m["content"] = new
            changed = True
    if "prompt" in rec and srec.get("prompt"):
        new, status = fix_text(rec["prompt"], srec["prompt"])
        st["prompt_" + status] += 1
        if status in ("reconstructed", "partial") and new != rec["prompt"]:
            rec["prompt"] = new
            changed = True
    return changed


def load_source(parquet: str) -> dict:
    cols = ["prompt_id", "messages"]
    df = pd.read_parquet(parquet)
    has_prompt = "prompt" in df.columns
    src = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        src[d["prompt_id"]] = {
            "messages": list(d["messages"]),
            "prompt": d.get("prompt") if has_prompt else None,
        }
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source parquet (English, by prompt_id)")
    ap.add_argument("--in", dest="inp", required=True, help="translated jsonl")
    ap.add_argument("--out", default=None, help="write repaired jsonl here")
    ap.add_argument("--report", action="store_true", help="measure only, print examples")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    src = load_source(args.src)
    st = Counter()
    n_records = n_changed = 0
    examples = []
    outfh = open(args.out, "w", encoding="utf-8") if args.out else None

    for line in open(args.inp, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        n_records += 1
        srec = src.get(rec.get("prompt_id"))
        if srec is not None:
            before_msgs = [m["content"] for m in rec.get("messages", [])] if args.report else None
            changed = fix_record(rec, srec, st)
            if changed:
                n_changed += 1
                if args.report and len(examples) < args.examples:
                    for i, m in enumerate(rec["messages"]):
                        if before_msgs and i < len(before_msgs) and m["content"] != before_msgs[i]:
                            examples.append((rec["prompt_id"], i, before_msgs[i], m["content"]))
                            break
        if outfh:
            outfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if outfh:
        outfh.close()

    print(f"records              = {n_records:,}")
    print(f"records changed      = {n_changed:,} ({100*n_changed/max(n_records,1):.2f}%)")
    print("message-level status :")
    for k, v in sorted(st.items(), key=lambda x: -x[1]):
        print(f"   {k:24} {v:,}")
    rec_ok = st.get("reconstructed", 0)
    part = st.get("partial", 0)
    masked = rec_ok + part + st.get("skipped_no_tail", 0)
    if masked:
        print(f"\nof masked messages: reconstructed={rec_ok:,} "
              f"({100*rec_ok/masked:.1f}%) partial={part:,} "
              f"({100*part/masked:.1f}%) skipped={st.get('skipped_no_tail',0):,}")

    for pid, i, before, after in examples:
        print(f"\n=== {pid[:10]} msg{i} ===")
        print(f"  BEFORE: {before[:260]!r}")
        print(f"  AFTER : {after[:260]!r}")


if __name__ == "__main__":
    main()
