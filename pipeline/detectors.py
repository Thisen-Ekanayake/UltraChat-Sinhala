"""Compiled content detectors shared by the content-feature and risk stages.

Each detector flags a textual phenomenon that is known to interact badly with
machine translation in general, and with low-resource (Sinhala) MT in
particular. Centralising the patterns here guarantees stage 3 (prevalence) and
stage 5 (risk register) measure exactly the same thing.
"""
from __future__ import annotations

import re

# Order matters only for human-readable reporting, not correctness.
DETECTORS: dict[str, re.Pattern] = {
    # Fenced code blocks ```...``` — MT will translate identifiers/keywords.
    "code_fence": re.compile(r"```"),
    # Inline code `x` — same risk at token granularity.
    "inline_code": re.compile(r"`[^`\n]+`"),
    # URLs / domains — must never be translated.
    "url": re.compile(r"https?://|www\.\w"),
    # Email addresses — entity to preserve.
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    # Markdown tables — pipe-delimited rows; structure easily corrupted.
    "markdown_table": re.compile(r"^\s*\|.*\|\s*$", re.M),
    # Bullet lists.
    "bullet_list": re.compile(r"^\s*[-*+]\s+\S", re.M),
    # Numbered lists.
    "numbered_list": re.compile(r"^\s*\d+[.)]\s+\S", re.M),
    # ATX headings.
    "heading": re.compile(r"^\s*#{1,6}\s+\S", re.M),
    # LaTeX / math — destroyed by MT. UNAMBIGUOUS signals only, because in a
    # code-heavy corpus the generic TeX delimiters \( \[ \d and bare $...$ are
    # indistinguishable from regex escapes, Swift/Kotlin string interpolation
    # ("\(x)") and shell/PHP variables ("$x"). We therefore match only:
    #   * named math control sequences (\frac, \sum, \alpha, ... — no
    #     programming language uses these);
    #   * \begin{ / \end{ environments;
    #   * display math $$...$$;
    #   * inline $...$ with a *braced* sub/superscript (x^{2}, a_{i}).
    # Trade-off (documented): bare inline math such as "$E=mc^2$" or "\(x+y\)"
    # is not counted. Such notation is rare here, and when it co-occurs with
    # code it is still masked by the code detectors.
    "latex_math": re.compile(
        r"\\(?:frac|sum|prod|int|sqrt|lim|infty|partial|nabla"
        r"|alpha|beta|gamma|delta|theta|lambda|mu|sigma|pi|cdot|times"
        r"|leq|geq|neq|approx|equiv|rightarrow|forall|exists)\b"
        r"|\\begin\{|\\end\{"
        r"|\$\$[^$]+\$\$"
        r"|\$(?!\()[^$\n]*[\^_]\{[^$\n]*\$"
    ),
    # HTML tags — markup to preserve.
    "html_tag": re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(\s[^>]*)?>"),
    # Emoji / pictographs.
    "emoji": re.compile(
        "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"),
    # Markdown links [text](url).
    "md_link": re.compile(r"\[[^\]]+\]\([^)]+\)"),
    # Bold/italic emphasis markers.
    "emphasis": re.compile(r"(\*\*|__|\*[^*\s]|_[^_\s])"),
}

# Severity for the Sinhala MT risk register (stage 5).
SEVERITY: dict[str, str] = {
    "code_fence": "high",
    "inline_code": "high",
    "latex_math": "high",
    "url": "high",
    "email": "medium",
    "html_tag": "medium",
    "markdown_table": "medium",
    "md_link": "medium",
    "numbered_list": "low",
    "bullet_list": "low",
    "heading": "low",
    "emphasis": "low",
    "emoji": "low",
}

# One-line rationale per detector for the report.
RATIONALE: dict[str, str] = {
    "code_fence": "MT translates code keywords/identifiers and string literals, breaking syntax; mask before translating.",
    "inline_code": "Inline technical tokens get translated or transliterated; protect with placeholders.",
    "latex_math": "LaTeX/math notation (TeX commands, $$display$$, $x^2$) is reordered/garbled by NMT; mask spans before translating. Excludes shell $(...) and currency.",
    "url": "URLs must be preserved verbatim; MT may percent-mangle or translate path segments.",
    "email": "Email addresses are entities that must survive untouched.",
    "html_tag": "Markup tags can be dropped or reordered, corrupting structure.",
    "markdown_table": "Pipe/column alignment is easily lost; cell boundaries shift after translation.",
    "md_link": "Link targets must be preserved while only display text is translated.",
    "numbered_list": "Item segmentation can shift; ordering must be retained across turns.",
    "bullet_list": "List markers usually survive but item boundaries can merge.",
    "heading": "Heading markers usually survive; included for completeness.",
    "emphasis": "Emphasis spans may be lost; low impact on meaning.",
    "emoji": "Usually passed through; included for completeness.",
}


def detect(text: str) -> dict[str, bool]:
    """Return {detector_name: present?} for one message."""
    return {name: bool(pat.search(text)) for name, pat in DETECTORS.items()}
