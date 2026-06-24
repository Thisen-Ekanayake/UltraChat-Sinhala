# Repairing ZWJ-loss in NLLB Sinhala machine translation

*Dataset-preparation methods note for the Sinhala UltraChat translation (for SinLLaMA SFT).*
Reproducible artifacts: [`tools/fix_sinhala_zwj.py`](../tools/fix_sinhala_zwj.py).

## 1. The problem

Sinhala writes consonant conjuncts that require an explicit **Zero-Width Joiner
(ZWJ, `U+200D`)** between the *al-lakuna* / hal sign (`U+0DCA`) and the following
consonant. For example the cluster **ප්‍ර** (*pra*) must be encoded as:

```
U+0DB4 (ප)  U+0DCA (්)  U+200D (ZWJ)  U+0DBB (ර)
```

NLLB-200's output for Sinhala **drops the ZWJ and emits a literal space
(`U+0020`)** in its place, so the model writes `ප් ර` (`0DB4 0DCA 0020 0DBB`)
instead of `ප්‍ර`. The damage is systematic and affects every ZWJ conjunct
family we observed:

| family | corrupted | correct | example |
|---|---|---|---|
| ra-conjunct (`C්‍ර`) | `ප් ර` | `ප්‍ර` | ප්‍රධාන (principal) |
| yansaya (`C්‍ය`) | `සෞඛ් ය` | `සෞඛ්‍ය` | සෞඛ්‍ය (health) |
| `C්‍ව`, `C්‍ෂ`, repaya `ර්‍C` | likewise | | මාධ්‍ය, ක්‍ෂ … |

**Root cause (measured).** The damage is done by NLLB's SentencePiece
**normalizer at encode time**, not by the model. Directly:

```
tok.backend_tokenizer.normalizer.normalize_str("්‍ර")  ->  "් ර"   (0DCA 0020 0DBB)
"ප්‍රධාන"  --encode→decode-->  "ප් රධාන"   (U+200D: 1 → 0)
"ද්‍රව්‍ය" --encode→decode-->  "ද් රව් ය"   (U+200D: 2 → 0)
"ග්ලූටන් රහිත" (real space)   --round-trip-->  unchanged   (U+200D: 0 → 0)
```

The normalizer rewrites `U+0DCA U+200D <C>` to `U+0DCA U+0020 <C>` *before*
tokenisation, so the ZWJ never reaches the model, the model never learns to emit
it, and **it cannot be recovered from the generated token ids**. The only fix is
to restore it orthographically on the decoded text — which is what this method
does, applied live in the translation stage (§6) so output is correct at source.

**Why it matters.** The space (a) breaks rendering — `ප් ර` shows the two
glyphs detached instead of the *pra* ligature; (b) changes tokenization — a
single word is split into two subword streams; and (c) degrades any model
trained on the data, since the conjunct vocabulary is fractured. For a published
SFT corpus this must be repaired.

## 2. Why it is not a find-and-replace

At the byte level `hal + space + letter` is **ambiguous**: it is *also* a
legitimate word boundary. In the 200-conversation sample, `hal + space` occurs
**43,772** times, followed by many different letters (ර 7,420; ය 4,854; ව 4,134;
ස 3,904; ක 3,864; …). A blind merge of, say, every `් ර` would wrongly fuse
real two-word sequences:

```
ග්ලූටන් රහිත   "gluten-free"  = ග්ලූටන් (gluten) + රහිත (without)   -> must stay split
එක් රටක්       "one country"  = එක් (one) + රටක් (a country)        -> must stay split
```

The two cases are indistinguishable without **lexical** knowledge of whether the
joined string is a real Sinhala word.

## 3. Method: lexicon-gated, ZWJ-aware merging

### Resources
- **Lexicon corpus** — `CPT-Dataset.txt`: a manually ZWJ-verified Sinhala corpus
  (~180M tokens; **145.2M** Sinhala word-tokens, **1,426,519** types, of which
  **178,430** contain `U+200D`). A word = a maximal run of Sinhala letters/marks
  + ZWJ, so conjunct words stay whole and spaces/punctuation split tokens.
- **Tokenizer** — `tokenizer/unigram_32000_0.9995` (SentencePiece unigram, 32k):
  its vocabulary already contains **2,652** ZWJ-bearing pieces (`▁ප්‍රධාන`,
  `▁සෞඛ්‍ය`, `▁ක්‍රීඩා`, …), used as a secondary oracle.

### Decision (per `hal + space` site)
For a candidate `L<space>R` (L ends in `U+0DCA`), form the ZWJ-join `L‍R` and
decide with three confidence tiers:

| tier | rule | confidence |
|---|---|---|
| **T1** corpus | `freq(L‍R)` in lexicon ≥ **5** | high |
| **T2** vocab | `L‍R` is a SentencePiece ZWJ piece | high |
| **T3** structural | `L` is a *lone* consonant+hal (e.g. `ක්`,`ග්`,`බ්`), `R` starts with `ර`/`ය`, **and** the cluster `L‍R[0]` is attested in the corpus | high — a lone hal-consonant is never a standalone word |
| — keep | otherwise | treated as a real word boundary |

- **T3 is restricted to `ර`/`ය`** (ra-/ya-conjuncts) **and gated on attested
  clusters**: the 4-char cluster (e.g. `ක්‍ර`, `ජ්‍ය`) must actually occur in the
  corpus, which blocks the heuristic from inventing non-clusters such as `ය්‍ර`.
  It recovers loanwords and rare inflections absent from the corpus as whole
  words (cranberry `ක්‍රන්බෙරි`, Grinch `ග්‍රින්ච්`, geometry `ජ්‍යාමිතීන්`).
  `ව`/`ෂ` are deliberately **not** in T3, because `lone-hal + ව` is overwhelmingly
  a real boundary in the data (`ක් වන`, `ක් විතර`); they stay corpus-gated.
- **Multi-conjunct words** (`ද්‍රව්‍ය`, `බ්‍රිතාන්‍ය`, `ප්‍රජාතන්ත්‍රවාදය`) are
  handled by joining a **maximal chain** of consecutive `hal+space` joints and
  taking the longest validated merge, left to right.

Only a single `U+0020` is ever replaced by `U+200D` inside a JSON string value,
so JSONL structure is preserved (verified after every run).

### Validation of the discriminator
The rule was checked against both oracles before any data was modified:

| candidate | vocab | corpus freq | decision |
|---|---|---|---|
| ප්‍රධාන | ✓ | 221,526 | merge |
| ක්‍රියාත්මක | ✓ | 142,980 | merge |
| සෞඛ්‍යය | ✓ | 2,926 | merge |
| **ග්ලූටන්‍රහිත** | ✗ | **0** | **keep split** |
| **එක්‍රටක්** | ✗ | **0** | **keep split** |

## 4. Result on `samples/part_01.first200.sinhala.jsonl`

| metric | value |
|---|---|
| `hal+space+letter` sites | 41,374 |
| **ZWJ merges applied** | **11,300** (T1 = 11,067, T3 = 233) |
| distinct word-forms merged | 1,351 |
| kept (real boundaries) | 30,074 |
| ZWJ `U+200D` count | 4 → **11,304** |
| residual unmerged lone-hal | 267 (88 distinct) |
| JSON validity | all lines valid |

The residual splits into (a) a *separate, out-of-scope* artifact — a stray `ක්`
token before a function word (`ක් පමණ`, `ක් හෝ`, `ක් සහ`), which is **not** a
ZWJ conjunct and is correctly left untouched; and (b) rare inflected
multi-conjunct forms whose stem is known but whose full inflection is below
threshold and whose prefix is not a lone hal (`අමුද් රව් යයන්` →
`අමුද්‍රව්‍යයන්`, ~2 sites). See §8.

## 5. Validation

A committed suite ([`tools/test_fix_sinhala_zwj.py`](../tools/test_fix_sinhala_zwj.py),
26 checks, all passing) makes the guarantees verifiable:

- **Unit tests** on adversarial strings exercise every branch — ra/ya/multi-conjunct
  merges, kept boundaries (`ග්ලූටන් රහිත`, `එක් රටක්`), the stray-`ක්` artifact,
  the cluster-gate (`ය්‍ර` is *not* invented), and idempotency on already-correct
  text.
- **Surgical-diff invariant** on the applied file vs its `.bak`: the two strings
  are identical in length and differ at *exactly* the 11,300 changed positions,
  each of which is `U+0020 → U+200D` and nothing else (equivalently,
  `fixed.replace(ZWJ,' ') == bak.replace(ZWJ,' ')`). No character is added,
  deleted, reordered, or otherwise altered; pre-existing ZWJs are preserved.
- **Idempotency**: re-running the repair on the output produces zero changes.
- **Structure**: identical line count, all lines valid JSON, identical keys,
  values identical after ZWJ-normalisation.
- **Consistency**: the finalized method re-applied to the `.bak` reproduces the
  on-disk file byte-for-byte.

## 6. Upstream fix (translation stage)

Because the ZWJ cannot be recovered downstream, the restoration is applied
**inside the translation stage, immediately after decoding**, so every written
record is correct and no separate repair pass over the released files is needed.

- [`pipeline/translation/sinhala_normalize.py`](../pipeline/translation/sinhala_normalize.py)
  is the canonical implementation (the batch tool in §7 imports the same core).
  `SinhalaZwjNormalizer.from_config()` loads the lexicon (cached) + vocab
  once; `.normalize(text)` is a cheap per-segment regex pass.
- [`pipeline/translation/translate.py`](../pipeline/translation/translate.py)
  builds the normalizer in `NllbTranslator.__init__` and applies it to every
  decoded batch right after `batch_decode` (before reassembly/unmasking).
- Config knobs (`pipeline/config.py`): `UC_ZWJ_FIX` (default on),
  `UC_ZWJ_CORPUS`, `UC_ZWJ_VOCAB`, `UC_ZWJ_LEXICON_CACHE`, `UC_ZWJ_MIN_FREQ`.
  A missing lexicon degrades gracefully to a warning (raw NLLB output), never a
  crash.

**End-to-end verification.** Pushing reference Sinhala through NLLB's own
tokenizer (which strips the ZWJ exactly as the model does) and then through the
in-pipeline normalizer restores it to a byte-exact match of the reference, while
a genuine boundary (`ග්ලූටන් රහිත`) is left untouched:

```
ref  : ශ්‍රී ලංකාවේ ප්‍රධාන සෞඛ්‍ය ක්‍රමය
nllb : ශ් රී ලංකාවේ ප් රධාන සෞඛ් ය ක් රමය      (ZWJ 4 → 0)
fixed: ශ්‍රී ලංකාවේ ප්‍රධාන සෞඛ්‍ය ක්‍රමය      (ZWJ → 4, exact match)
```

## 7. Reproducing

```bash
# dry-run (report only), building/caching the lexicon from the corpus
python3 tools/fix_sinhala_zwj.py samples/part_01.first200.sinhala.jsonl \
    --corpus CPT-Dataset.txt \
    --vocab  tokenizer/unigram_32000_0.9995.vocab \
    --lexicon-cache /path/to/sinhala_lexicon.pkl \
    --threshold 5 \
    --report results/audit_part_01.md

# apply in place (keeps a .bak; aborts if any output line is invalid JSON)
python3 tools/fix_sinhala_zwj.py samples/part_01.first200.sinhala.jsonl ... --apply
```

## 8. Limitations & next steps
- **Out-of-scope spacing artifacts.** A separate defect inserts spurious spaces
  inside *non-ZWJ* clusters (e.g. `මත් ද්‍රව්‍ය` for `මත්ද්‍රව්‍ය`) and produces
  stray `ක්` tokens. These are not ZWJ losses and are intentionally not touched.
- **Sub-threshold inflected multi-conjunct forms** are left split (≈1% of
  multi-conjunct instances in the sample). A safe **stem-prefix** extension —
  accept `L‍R` when a known word is a prefix that covers all inserted ZWJs — would
  close this; it must be validated against the gluten-free false-positive class
  before use on published data.
- **Upstream fix — done (§6).** Restoration runs live in the translation stage,
  so newly translated data is correct at source. (The ZWJ is unrecoverable
  downstream, so this is restoration on the decoded text, not a tokenizer patch.)
- **Roll-out.** Apply the batch tool to any pre-existing translated shards before
  release; report per-shard merge counts and JSON validity as a data-quality table.
