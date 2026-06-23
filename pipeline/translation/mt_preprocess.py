"""Pre-/post-processing for English->Sinhala machine translation (stage 9).

Two jobs, both flagged as mandatory by the corpus analysis (report §6/§8):

1. **Mask** spans that MT must not touch — fenced code, inline code, URLs,
   e-mail addresses, HTML tags, markdown links and LaTeX/math — by replacing
   each with a unique placeholder, translating around them, then **restoring**
   the originals. The detector set mirrors ``detectors.py`` (the same patterns
   the analysis measured as risky), with full-span variants where masking needs
   the whole span rather than just a presence signal.

2. **Segment** messages into sentence-sized pieces so each translation request
   stays under NLLB's 512-token positional limit without truncating
   mid-sentence (report §8.4). Newline structure (lists, paragraphs) is
   preserved exactly on reassembly — only the text of each line is translated.

The placeholder is the mathematical white bracket pair ``⟦n⟧``: rare in the
corpus, stable through SentencePiece, and the restore step is tolerant — it
matches ``⟦ n ⟧`` even if the model injects spaces, and any placeholder the
model drops is re-appended rather than silently lost. Fidelity should still be
spot-checked on a sample before a full run.
"""
from __future__ import annotations

import re

from pipeline.detectors import DETECTORS

from pipeline import config

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
# Full-span variants for masking (detectors.py carries presence-only versions).
_FENCE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_DISPLAY_MATH = re.compile(r"\$\$[^$]+\$\$")
_URL_FULL = re.compile(r"https?://\S+|www\.\S+")

# OUTER-FIRST priority: a fenced block is claimed before its inner backticks
# read as inline code; a markdown link before the bare URL inside it; etc.
_PROTECT: list[tuple[str, re.Pattern]] = [
    ("code_fence", _FENCE_BLOCK),
    ("md_link", DETECTORS["md_link"]),
    ("url", _URL_FULL),
    ("email", DETECTORS["email"]),
    ("display_math", _DISPLAY_MATH),
    ("latex_math", DETECTORS["latex_math"]),
    ("html_tag", DETECTORS["html_tag"]),
    ("inline_code", DETECTORS["inline_code"]),
]

_PLACEHOLDER = "⟦{}⟧"
_PLACEHOLDER_RE = re.compile(r"⟦\s*(\d+)\s*⟧?")  # tolerant: allows spaces / missing ⟧


def protect(text: str) -> tuple[str, list[str]]:
    """Replace must-preserve spans with ``⟦n⟧`` placeholders.

    Returns the masked text and the ordered list of original spans (list index
    == placeholder number).
    """
    claimed: list[tuple[int, int]] = []
    for _name, pat in _PROTECT:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if s == e or any(s < ce and cs < e for cs, ce in claimed):
                continue
            claimed.append((s, e))
    if not claimed:
        return text, []

    claimed.sort()
    originals: list[str] = []
    out: list[str] = []
    cursor = 0
    for s, e in claimed:
        out.append(text[cursor:s])
        out.append(_PLACEHOLDER.format(len(originals)))
        originals.append(text[s:e])
        cursor = e
    out.append(text[cursor:])
    return "".join(out), originals


def restore(text: str, originals: list[str]) -> str:
    """Put masked spans back. Tolerant of model-injected spaces; any placeholder
    the model dropped is appended so its content is never silently lost."""
    if not originals:
        return text
    seen: set[int] = set()

    def _sub(m: re.Match) -> str:
        idx = int(m.group(1))
        if 0 <= idx < len(originals):
            seen.add(idx)
            return originals[idx]
        return m.group(0)

    restored = _PLACEHOLDER_RE.sub(_sub, text)
    missing = [originals[i] for i in range(len(originals)) if i not in seen]
    if missing:
        restored = restored.rstrip() + " " + " ".join(missing)
    return restored


# ---------------------------------------------------------------------------
# Sentence segmentation (newline structure preserved on reassembly)
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def _nltk_splitter():
    """Return an nltk sentence splitter if a punkt model is available.

    nltk >= 3.9 ships the tokenizer as ``punkt_tab``; older versions as
    ``punkt``. Accept either; fall back to the regex splitter if neither is
    installed (segmentation still works, just more coarsely).
    """
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        for res in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(res)
                return sent_tokenize
            except LookupError:
                continue
        return None
    except Exception:
        return None


_SENT_TOKENIZE = _nltk_splitter()


def _hard_wrap(piece: str, limit: int) -> list[str]:
    """Split an over-long sentence on whitespace so no piece exceeds ``limit``
    characters (a coarse proxy for NLLB's 512-token input limit)."""
    if len(piece) <= limit:
        return [piece]
    out, cur = [], ""
    for word in piece.split(" "):
        if cur and len(cur) + 1 + len(word) > limit:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    final: list[str] = []
    for p in out:                      # a single word longer than the limit:
        while len(p) > limit:          # chop by character as a last resort.
            final.append(p[:limit])
            p = p[limit:]
        if p:
            final.append(p)
    return final


def _split_line(line: str, limit: int) -> list[str]:
    sents = _SENT_TOKENIZE(line) if _SENT_TOKENIZE else _SENT_SPLIT.split(line)
    out: list[str] = []
    for s in sents:
        if s.strip():
            out.extend(_hard_wrap(s, limit))
    return out


def to_segments(text: str, limit: int | None = None):
    """Split ``text`` into translation-sized segments, preserving line layout.

    Returns ``(segments, layout)`` where ``segments`` is the flat list of
    non-empty sentence pieces to translate (in order) and ``layout`` is an
    opaque object consumed by :func:`from_segments` to rebuild the message with
    its original newlines intact.
    """
    limit = limit or config.MAX_SEGMENT_CHARS
    lines = text.split("\n")
    segments: list[str] = []
    per_line: list[int] = []           # sentence-segments contributed by each line
    for line in lines:
        if not line.strip():
            per_line.append(0)         # blank/whitespace line -> kept verbatim
            continue
        segs = _split_line(line, limit)
        per_line.append(len(segs))
        segments.extend(segs)
    return segments, (lines, per_line)


def from_segments(translated: list[str], layout) -> str:
    """Rebuild a message from its translated segments and the saved layout."""
    lines, per_line = layout
    out: list[str] = []
    i = 0
    for orig_line, n in zip(lines, per_line):
        if n == 0:
            out.append(orig_line)      # blank line: passthrough
        else:
            out.append(" ".join(translated[i:i + n]))
            i += n
    return "\n".join(out)
