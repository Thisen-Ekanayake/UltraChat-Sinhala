# MT Masking-Sentinel Leakage — Problem & Fix

_Dataset-prep artifact for UltraChat-Sinhala. Investigated and repaired 2026-06-26._

## TL;DR

The machine-translation pipeline masks spans that must not be translated (code,
URLs, math, markup) with placeholders, then restores them after translation. The
placeholder used a pair of rare math brackets that **NLLB's SentencePiece strips
away**, leaving a bare digit the restore step could not recognise. As a result
every masked span was **displaced to the end of its message** with a **stray
digit left inline**. This affected **~2.56 % of messages** (≈100 % of messages
that contained any masked span). We repaired the published data in place by
re-restoring each affected message from its English source, and hardened the
pipeline so the failure cannot recur.

---

## 1. The problem

### 1.1 How masking is supposed to work
[`pipeline/translation/mt_preprocess.py`](../pipeline/translation/mt_preprocess.py)
masks must-preserve spans before translation and restores them after:

- `protect(text)` replaces each detected span (fenced/inline code, URLs, emails,
  HTML, LaTeX/math, markdown links) with the placeholder `⟦n⟧` and returns the
  ordered list of originals (`originals[n]` ↔ placeholder `n`).
- The translator masks the **whole message**, segments it, translates, reassembles
  (`from_segments` preserves the line layout), then calls `restore()`.
- `restore(text, originals)` substitutes each `⟦n⟧` back to `originals[n]`.

The placeholder is `⟦n⟧`, built from **U+27E6 / U+27E7** (mathematical white
brackets), chosen because they are rare in the corpus.

### 1.2 The bug
NLLB-200's SentencePiece normaliser **strips U+27E6/U+27E7** during translation,
so `⟦0⟧` comes back as a bare **`0`**. But `restore`'s matcher required the
opening bracket:

```python
_PLACEHOLDER_RE = re.compile(r"⟦\s*(\d+)\s*⟧?")   # needs the ⟦
```

With the `⟦` gone the regex never matches, so the index is treated as **missing**
and the original span is **appended at the end of the message**, while the bare
digit stays where the placeholder was:

```python
missing = [originals[i] for i in range(len(originals)) if i not in seen]
if missing:
    restored = restored.rstrip() + " " + " ".join(missing)   # displaced to end
```

**Two visible artifacts per leaked placeholder:** (a) a stray bare digit inline,
(b) the real span (code/URL/…) relocated to the message end.

### 1.3 Incidence (measured)
Aligning every translated message to its English source by `prompt_id` and
re-running `protect` on the source (part_01, 146,216 messages):

| signal | count | of masked |
|---|---:|---:|
| messages containing a masked span | 3,748 (2.56 % of all messages) | — |
| original span appended at message end (leak) | 3,748 | **100.0 %** |
| stray bare digit left inline | 3,563 | 95.1 % |

Placeholders per message ranged from 1 (most common) to 23+. **Every** masked
message leaked — the bracket-stripping is deterministic.

---

## 2. The fix

### 2.1 Data repair (existing published data)
[`tools/fix_masking_leak.py`](../tools/fix_masking_leak.py) re-restores each
affected message **from the English source**, putting every span back where it
belongs rather than at the end. It exploits the fact that `from_segments`
preserves line structure, so the translated body line-aligns with the masked
source:

1. `protect(source)` → the exact original spans `O` **and** the masked text, which
   records *which placeholder index sits on which source line*.
2. Strip the appended tail (`" ".join(O)`) that `restore` added at the message end.
3. Line-align the body with the masked source; on each line, replace the leaked
   bare digit for the index(es) known to belong there with the original span.
   Matching is by **exact digit string, constrained to the indices expected on
   that line** — so real numbers elsewhere (list markers `1. 2. 3.`, "divisible
   by **4**", "**8x** traffic") are never touched.
4. **Validate:** every span must be reinserted exactly once. Any span that cannot
   be located (NLLB dropped or hallucinated its digit) is **re-appended, never
   dropped**; that message is counted *partial*. Messages without the append
   signature are left byte-for-byte unchanged.

### 2.2 Pipeline hardening (prevent recurrence)
`restore()` in `mt_preprocess.py` is now **three-tier** so the same SentencePiece
behaviour self-heals in future runs:

1. Match the intact `⟦n⟧` (as before).
2. **Bracket-stripped fallback:** recover each still-missing index from its first
   standalone *bare-digit* occurrence, ascending, using NUL sentinels so a digit
   *inside* an already-mapped span is never re-matched.
3. Append only what truly cannot be located (content is never lost).

Round-trip `protect → restore` is verified, and bare-digit / missing cases are
covered by inline tests.

---

## 3. Results

Applied to all 20 shards (10 SFT + 10 GEN):

| | reconstructed in place | partial (safe re-append) | reconstruction rate |
|---|---:|---:|---:|
| **SFT** messages | 32,002 | 6,193 | 83.8 % |
| **GEN** messages | 24,424 | 5,190 | 82.5 % |
| **Both** messages | **56,426** | 11,383 | **83.2 %** |
| Prompts | 6,248 | 2,779 | 69.2 % |

- **37,125 records changed** (of 515,311); the rest contained no masked spans.
- **Validation (part_01):** append-displacement signature fell **100 % → 24 %**
  (the residual 24 % are *partial* messages that safely re-append the few
  unplaceable spans); **0 rows lost**, **0 invalid JSON lines**, line counts
  identical to the inputs.

### Before → after (real examples)
```
URL:    …වෙබ් අඩවිය 0 ඔවුන්ගේ…        →  …වෙබ් අඩවිය www.euracoustics.org. ඔවුන්ගේ…
code:   …උදාහරණ වැඩසටහනක්.\n\n0\n\n…   →  …උදාහරණ වැඩසටහනක්.\n\n```java\nimport java.io…```…
inline: - 1 නූල් දෙක…  - 2 …  - 3 …    →  - `concat` නූල් දෙක…  - `distinct` …  …
```
Real numbers in the same messages (`8x`, `2060`, "divisible by 4") are preserved.

The repaired shards were uploaded to `gs://sinllama-cpt/UltraChat-Sinhala/{sft,gen}/cleaned/`
(overwriting the ZWJ-cleaned-but-leaked versions; raw remains in `…/uncleaned/`
and `gs://sinllama-cpt/UltraChat-split/`).

---

## 4. Limitations

- **Partial messages (16.8 %):** when NLLB dropped or hallucinated a placeholder's
  digit (so it cannot be located on its line), that span is re-appended at the
  message end rather than placed in line. No content is lost, and any spans that
  *were* located are still placed correctly — so partials are strictly better than
  the original, just not perfect.
- **Real-number collision (rare):** if a line genuinely contains a single-digit
  number equal to a placeholder index that also belonged on that line, the
  matcher could replace the real number. Bounded by the line constraint and the
  small index range; not observed in spot checks.
- This repair is **source-guided** (joins to the English shards by `prompt_id`);
  it cannot run without them. The pipeline hardening (§2.2) needs no source and
  prevents the problem at translation time going forward.
