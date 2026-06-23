#!/usr/bin/env python3
"""Stage 6 — aggregate all stage outputs into a single academic markdown report.

Reads the JSON produced by stages 1-5 and emits a structured, citation-ready
report (results/ANALYSIS_REPORT.md) covering dataset overview, methodology,
results (structure, volume, content features, multilingual composition),
a prioritised Sinhala-translation risk register, a translation-cost analysis,
discussion/recommendations, limitations and reproducibility notes.

This stage performs no data passes; it is pure presentation over the JSON
artifacts, so the report is fully reproducible from the intermediate files.
"""
from __future__ import annotations

import sys
import time

import config
from uc_common import fmt_int, get_logger, load_result, result_path

log = get_logger("report")
REPORT = "ANALYSIS_REPORT.md"


# ---------------------------------------------------------------------------
# Small markdown helpers
# ---------------------------------------------------------------------------
def h(level: int, text: str) -> str:
    return f"\n{'#' * level} {text}\n"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def usd(x: float) -> str:
    return f"${x:,.2f}"


def safe_load(name: str):
    try:
        return load_result(name)
    except Exception as exc:  # pragma: no cover
        log.warning("Missing %s (%s)", name, exc)
        return None


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def sec_header() -> str:
    return (
        f"# Corpus Analysis of UltraChat 200k for English→Sinhala "
        f"Machine Translation\n\n"
        f"*Pre-translation dataset characterisation and risk assessment.*\n\n"
        f"- **Dataset:** `{config.HF_REPO}`\n"
        f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"- **Pipeline:** modular streaming analysis (stages 0–6)\n"
    )


def sec_abstract(struct, text) -> str:
    body = h(2, "Abstract")
    if not (struct and text):
        return body + "_Insufficient data to generate abstract._\n"
    o_s = struct["overall"]
    o_t = text["overall"]
    body += (
        f"This report characterises the UltraChat 200k corpus prior to an "
        f"English→Sinhala translation effort. The analysed material comprises "
        f"**{fmt_int(o_s['n_records'])} dialogues** and "
        f"**{fmt_int(o_s['n_messages'])} messages** totalling "
        f"**{fmt_int(o_t['total_chars'])} translatable characters** "
        f"(~{fmt_int(o_t['approx_total_tokens'])} tokens, {o_t['token_method']}). "
        f"We quantify conversational structure, text-volume distributions, the "
        f"prevalence of machine-translation–sensitive content (code, markup, "
        f"mathematics, URLs), and multilingual composition, and we derive a "
        f"character-billed translation-cost model. The findings inform a "
        f"prioritised risk register for low-resource Sinhala MT and the "
        f"feasibility of the proposed back-translation quality-estimation "
        f"design under a fixed compute budget.\n"
    )
    return body


def sec_methodology(text) -> str:
    body = h(2, "1. Methodology")
    method = text["overall"]["token_method"] if text else "n/a"
    body += (
        "The dataset is processed as parquet shards via a six-stage modular "
        "pipeline. Each analysis stage performs a single streaming pass "
        "(bounded memory, independent of corpus size) and writes a JSON "
        "artifact; this report is generated purely from those artifacts, so it "
        "is reproducible without re-reading the corpus.\n\n"
        "- **Stage 1 — Structure:** turn counts, role inventory, alternation "
        "validity, schema integrity.\n"
        "- **Stage 2 — Text volume:** exact character counts (the unit "
        "character-billed MT APIs charge on), length distributions, token "
        f"estimates ({method}), and a cost model.\n"
        "- **Stage 3 — Content features:** regex prevalence of code, inline "
        "code, URLs, tables, lists, headings, LaTeX/math, HTML, emoji, links, "
        "emphasis.\n"
        "- **Stage 4 — Unicode/multilingual:** non-ASCII volume and per-script "
        "composition (Unicode `\\p{Script=…}` properties).\n"
        "- **Stage 5 — Risk register:** synthesis into a prioritised "
        "English→Sinhala risk assessment with examples.\n\n"
        "Character counts are **exact**; token counts and monetary costs are "
        "**estimates** and are labelled as such.\n"
    )
    return body


def sec_structure(struct) -> str:
    body = h(2, "2. Conversational Structure")
    if not struct:
        return body + "_No structure data._\n"
    rows = []
    for split, s in struct["per_split"].items():
        t = s["turns_per_dialogue"]
        rows.append([
            f"`{split}`", fmt_int(s["n_records"]), fmt_int(s["n_messages"]),
            f"{t['min']:.0f}", f"{t['p50']:.0f}", f"{t['mean']:.1f}",
            f"{t['max']:.0f}", fmt_int(s["n_empty_messages"]),
            fmt_int(s["n_nonalternating_dialogues"]),
        ])
    body += table(
        ["Split", "Dialogues", "Messages", "Min turns", "Median", "Mean",
         "Max", "Empty msgs", "Non-alternating"],
        rows,
    )
    o = struct["overall"]
    body += (
        f"\nAcross all splits the corpus contains **{fmt_int(o['n_records'])} "
        f"dialogues** / **{fmt_int(o['n_messages'])} messages**, with "
        f"**{fmt_int(o['n_empty_messages'])} empty messages** and "
        f"**{fmt_int(o['n_nonalternating_dialogues'])} dialogues** violating "
        f"strict user/assistant alternation. The translation pipeline must "
        f"preserve `prompt_id`, message order and role labels when "
        f"reassembling translated dialogues.\n"
    )
    return body


def sec_volume(text) -> str:
    body = h(2, "3. Text Volume and Length Distributions")
    if not text:
        return body + "_No text data._\n"
    rows = []
    for split, s in text["per_split"].items():
        mc = s["message_chars"]
        rows.append([
            f"`{split}`", fmt_int(s["total_chars"]),
            fmt_int(s["approx_total_tokens"]),
            f"{mc['p50']:.0f}", f"{mc['p95']:.0f}", f"{mc['max']:.0f}",
            fmt_int(s["chars_by_role"].get("user", 0)),
            fmt_int(s["chars_by_role"].get("assistant", 0)),
        ])
    body += table(
        ["Split", "Total chars", "≈Tokens", "Median msg", "p95 msg",
         "Max msg", "User chars", "Assistant chars"],
        rows,
    )
    o = text["overall"]
    body += (
        f"\nTotal translatable volume: **{fmt_int(o['total_chars'])} characters** "
        f"(~{fmt_int(o['approx_total_tokens'])} tokens). Long messages (p95 and "
        f"beyond) exceed common single-request limits and require sentence-aware "
        f"chunking before translation.\n"
    )
    return body


def sec_content(content) -> str:
    body = h(2, "4. Machine-Translation–Sensitive Content")
    if not content:
        return body + "_No content-feature data._\n"
    o = content["overall"]
    rows = []
    for name, cnt in sorted(o["feature_counts"].items(), key=lambda kv: -kv[1]):
        rows.append([f"`{name}`", fmt_int(cnt), f"{o['feature_pct'][name]:.2f}%"])
    body += table(["Feature", "Messages", "% of messages"], rows)
    body += (
        f"\n**{fmt_int(o['n_messages_with_any_feature'])}** messages "
        f"(of {fmt_int(o['n_messages'])}) contain at least one such feature. "
        f"Code, inline code, LaTeX/math and URLs must be masked with "
        f"placeholders before translation and restored afterwards.\n"
    )
    return body


def sec_unicode(uni) -> str:
    body = h(2, "5. Multilingual Composition")
    if not uni:
        return body + "_No unicode data._\n"
    o = uni["overall"]
    body += (
        f"**{fmt_int(o['n_nonascii_messages'])}** messages contain non-ASCII "
        f"characters ({fmt_int(o['nonascii_chars'])} chars total, of which "
        f"{fmt_int(o['typographic_chars'])} are typographic and "
        f"{fmt_int(o['foreign_script_chars'])} are foreign-script letters). "
        f"**{fmt_int(o['n_foreign_script_messages'])}** messages carry genuinely "
        f"foreign-script text.\n\n"
    )
    scripts = o.get("script_message_counts", {})
    if scripts:
        rows = [[f"`{s}`", fmt_int(c)] for s, c in list(scripts.items())[:12]]
        body += table(["Script", "Messages"], rows)
    return body


def sec_risks(risks) -> str:
    body = h(2, "6. Sinhala Translation Risk Register")
    if not risks:
        return body + "_No risk data._\n"
    summ = risks["summary"]
    body += (
        f"**{fmt_int(summ['n_messages_high_severity'])}** messages "
        f"({summ['pct_messages_high_severity']:.2f}%) carry at least one "
        f"HIGH-severity feature requiring placeholder masking. Long-message "
        f"counts (chunking risk): "
        + ", ".join(f"`>{k}`: {fmt_int(v)}" for k, v in summ["messages_over_length"].items())
        + f"; longest message = {fmt_int(summ['max_message_chars'])} chars.\n\n"
    )
    rows = []
    for r in risks["risk_register"]:
        rows.append([
            f"`{r['category']}`", r["severity"].upper(),
            fmt_int(r["messages_affected"]), f"{r['pct_messages']:.2f}%",
            r["rationale"],
        ])
    body += table(["Category", "Severity", "Messages", "%", "Why it matters"], rows)

    fs = risks.get("foreign_script_risk") or {}
    if fs:
        body += (
            f"\n**Foreign-script source.** {fmt_int(fs.get('n_foreign_script_messages', 0))} "
            f"messages already contain non-Latin scripts; "
            f"{fmt_int(fs.get('messages_already_in_sinhala', 0))} already contain "
            f"Sinhala (risk of double-translation). {fs.get('note', '')}\n"
        )

    body += h(3, "6.1 Illustrative examples")
    for r in risks["risk_register"]:
        if r["severity"] != "high" or not r["examples"]:
            continue
        body += f"\n**`{r['category']}`** ({r['severity'].upper()}):\n"
        for ex in r["examples"]:
            body += f"> {ex}\n"
    return body


def sec_cost(text) -> str:
    body = h(2, "7. Translation-Cost Analysis")
    if not (text and "cost_model" in text):
        return body + "_No cost model._\n"
    cm = text["cost_model"]
    a = cm["assumptions"]
    body += (
        f"Assumptions: Google Cloud Translation at "
        f"**{usd(a['google_translate_usd_per_million_chars'])}/1M chars**, "
        f"GCP free-trial credit **{usd(a['gcp_free_trial_credit_usd'])}**, "
        f"Sinhala expansion factor **{a['sinhala_expansion_factor']}×** for the "
        f"back-translation leg. {a['note']}\n\n"
    )
    rows = []
    for split, b in cm["per_split"].items():
        rows.append([
            f"`{split}`", fmt_int(b["chars"]), usd(b["one_way_usd"]),
            usd(b["round_trip_usd"]), f"{b['round_trip_pct_of_credit']:.0f}%",
        ])
    rows.append(["**SFT only**", fmt_int(cm["sft_only"]["chars"]),
                 usd(cm["sft_only"]["one_way_usd"]),
                 usd(cm["sft_only"]["round_trip_usd"]),
                 f"{cm['sft_only']['round_trip_pct_of_credit']:.0f}%"])
    rows.append(["**Whole dataset**", fmt_int(cm["whole_dataset"]["chars"]),
                 usd(cm["whole_dataset"]["one_way_usd"]),
                 usd(cm["whole_dataset"]["round_trip_usd"]),
                 f"{cm['whole_dataset']['round_trip_pct_of_credit']:.0f}%"])
    body += table(
        ["Scope", "Chars", "One-way (en→si)", "Round-trip (en→si→en)",
         "Round-trip % of $300"],
        rows,
    )
    body += (
        f"\nThe **{usd(a['gcp_free_trial_credit_usd'])}** credit buys roughly "
        f"**{fmt_int(cm['chars_buyable_with_credit_one_way'])} characters** "
        f"one-way. The proposed round-trip design **doubles** the per-character "
        f"bill. As the table shows, the paid Cloud Translation API cannot cover "
        f"the full corpus within the free-trial credit by orders of magnitude.\n"
    )
    return body


def sec_discussion(risks, text) -> str:
    body = h(2, "8. Discussion and Recommendations")
    body += (
        "1. **Translator choice.** Character-billed cloud MT is financially "
        "infeasible at this volume on the free-trial credit. Running an "
        "open-source MT model (e.g. NLLB-200, `sin_Sinh`) on the VM GPU removes "
        "per-character billing and is the only budget-viable path for the full "
        "corpus.\n"
        "2. **Quality estimation.** Round-trip (en→si→en) translation through a "
        "single engine can mask adequacy errors (the engine may 'clean up' on "
        "the return leg) and doubles MT cost. A reference-free QE model "
        "(e.g. CometKiwi) scoring the source→Sinhala pair directly is cheaper "
        "and methodologically sounder.\n"
        "3. **Pre-processing is mandatory.** Mask code, inline code, LaTeX/math "
        "and URLs before translation; preserve markdown/table/list structure; "
        "restore afterwards.\n"
        "4. **Chunking.** Long messages must be split on sentence boundaries to "
        "respect request limits without harming coherence.\n"
        "5. **Filtering bias.** A fixed quality threshold (e.g. keep score > 85) "
        "preferentially discards long/technical/creative messages, shifting the "
        "SFT distribution; tune any threshold on a human-checked sample and "
        "consider stratified retention.\n"
        "6. **Engineering.** Checkpoint by `prompt_id` + message index for "
        "resumability across the full corpus, and preserve role order on "
        "reassembly.\n"
    )
    return body


def sec_repro() -> str:
    body = h(2, "9. Reproducibility")
    body += (
        "```\n"
        "bash run_all.sh            # download + all stages + this report\n"
        "bash run_all.sh --skip-download   # reuse existing UC_DATA_DIR\n"
        "```\n\n"
        f"Intermediate JSON artifacts and this report are written to "
        f"`{config.RESULTS_DIR}`. Stage scripts are independently runnable for "
        "debugging (`python 0X_*.py`).\n"
    )
    return body


def main() -> int:
    struct = safe_load("01_structure.json")
    text = safe_load("02_text_stats.json")
    content = safe_load("03_content_features.json")
    uni = safe_load("04_unicode_scripts.json")
    risks = safe_load("05_translation_risks.json")

    parts = [
        sec_header(),
        sec_abstract(struct, text),
        sec_methodology(text),
        sec_structure(struct),
        sec_volume(text),
        sec_content(content),
        sec_unicode(uni),
        sec_risks(risks),
        sec_cost(text),
        sec_discussion(risks, text),
        sec_repro(),
    ]
    path = result_path(REPORT)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    log.info("Wrote report: %s", path)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
