"""Central configuration for the UltraChat 200k analysis pipeline.

All paths can be overridden with environment variables so the same scripts run
unchanged on a local machine or a GCP VM:

    UC_DATA_DIR     directory holding the *.parquet shards (default: <repo>/data)
    UC_RESULTS_DIR  directory for intermediate JSON + the final report
                    (default: <analysis>/results)
    UC_HF_REPO      HuggingFace dataset repo id

This module is import-only; it performs no I/O beyond filesystem discovery.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This module lives at <repo>/pipeline/config.py; data/, results/ and the model
# directory live at the repo root, one level up from the package.
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent

DATA_DIR = Path(os.environ.get("UC_DATA_DIR", ROOT_DIR / "data")).resolve()
RESULTS_DIR = Path(os.environ.get("UC_RESULTS_DIR", ROOT_DIR / "results")).resolve()

# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------
HF_REPO = os.environ.get("UC_HF_REPO", "HuggingFaceH4/ultrachat_200k")
HF_REPO_TYPE = "dataset"

# Splits we expect to find. Discovery (below) is authoritative; this list is
# only used to order/label output and warn about missing splits.
EXPECTED_SPLITS = ["train_sft", "test_sft", "train_gen", "test_gen"]

# Split groups. The Sinhala translation effort targets SFT; GEN is kept for
# the tokenisation analysis (stage 7), which reports the two groups separately.
SFT_SPLITS = ["train_sft", "test_sft"]
GEN_SPLITS = ["train_gen", "test_gen"]
SPLIT_GROUPS = {"sft": SFT_SPLITS, "gen": GEN_SPLITS}


def split_group(split: str) -> str:
    """Return 'sft', 'gen' or 'other' for a split name."""
    for group, members in SPLIT_GROUPS.items():
        if split in members:
            return group
    return "other"


# ---------------------------------------------------------------------------
# Tokeniser (stage 7) — SinLlama merged model directory (HF format).
# Override on the VM with UC_TOKENIZER_DIR=/path/to/SinLlama_merged_bf16
# ---------------------------------------------------------------------------
TOKENIZER_DIR = Path(
    os.environ.get("UC_TOKENIZER_DIR", ROOT_DIR / "SinLlama_merged_bf16")
).resolve()

# Context-window thresholds (in tokens) to flag dialogues against, for SFT
# training budgeting. A full dialogue is one training sample.
CONTEXT_WINDOWS = [2048, 4096, 8192]

# ---------------------------------------------------------------------------
# Cost model constants (for the translation-cost section of the report)
# ---------------------------------------------------------------------------
# Google Cloud Translation (Basic/Advanced text) list price, USD per 1M chars.
# Verify against current pricing before quoting; recorded here for reproducibility.
GOOGLE_TRANSLATE_USD_PER_MCHAR = 20.0
GCP_FREE_TRIAL_CREDIT_USD = 300.0
# Rough multiplier: Sinhala output tends to be longer than English source in
# characters, so the back-translation leg bills on more characters than the
# forward leg. Used only for the round-trip cost illustration.
SINHALA_EXPANSION_FACTOR = 1.15

# Heuristic used when a real tokenizer is unavailable.
CHARS_PER_TOKEN_HEURISTIC = 4.0

# Streaming batch size for pyarrow iter_batches.
BATCH_SIZE = 2000

# ---------------------------------------------------------------------------
# Translation (stage 9) — NLLB-200 English -> Sinhala
# ---------------------------------------------------------------------------
# Open-source NMT is the only budget-viable path at this corpus volume (see the
# cost model / report §8). NLLB-200-3.3B covers sin_Sinh natively. All knobs are
# env-overridable so the same script runs on a CPU VM and on a GPU box.
NLLB_MODEL = os.environ.get("UC_NLLB_MODEL", "facebook/nllb-200-3.3B")
SRC_LANG = os.environ.get("UC_SRC_LANG", "eng_Latn")   # FLORES-200 code
TGT_LANG = os.environ.get("UC_TGT_LANG", "sin_Sinh")   # Sinhala

# Device / precision. "auto" picks cuda if available else cpu; on cpu the dtype
# defaults to bfloat16 (this VM's Xeon has AMX-BF16) which halves the weight
# footprint (3.3B params: ~13 GB fp32 -> ~6.6 GB bf16, fits 16 GB RAM).
TRANSLATE_DEVICE = os.environ.get("UC_DEVICE", "auto")  # auto | cuda | cpu
TRANSLATE_DTYPE = os.environ.get("UC_DTYPE", "auto")    # auto | fp32 | fp16 | bf16
# Number of torch CPU threads (0 = let torch decide). Set to the VM's vCPU count.
CPU_THREADS = int(os.environ.get("UC_CPU_THREADS", "0"))

# Batching / length limits. NLLB's positional limit is 512 tokens; long messages
# are sentence-segmented (see mt_preprocess) so each segment fits.
TRANSLATE_BATCH = int(os.environ.get("UC_TRANSLATE_BATCH", "16"))   # segments per fwd pass
DIALOGUE_CHUNK = int(os.environ.get("UC_DIALOGUE_CHUNK", "32"))     # dialogues buffered before a flush/checkpoint
NUM_BEAMS = int(os.environ.get("UC_NUM_BEAMS", "1"))               # 1 = greedy (fastest)
MAX_INPUT_TOKENS = int(os.environ.get("UC_MAX_INPUT_TOKENS", "512"))
MAX_NEW_TOKENS = int(os.environ.get("UC_MAX_NEW_TOKENS", "512"))
MAX_SEGMENT_CHARS = int(os.environ.get("UC_MAX_SEGMENT_CHARS", "1000"))  # hard wrap for segmentation

# Mask code/URLs/math/markup before translating and restore after (report §8.3).
MASK_BEFORE_TRANSLATE = os.environ.get("UC_MASK", "1") not in ("0", "false", "False")

# Post-decode Sinhala ZWJ restoration. NLLB's SentencePiece normaliser maps the
# conjunct joiner (U+0DCA U+200D <C>) to U+0DCA U+0020 <C>, destroying the
# Zero-Width Joiner at encode time (measured), so NLLB output never contains it.
# We restore it orthographically on each decoded segment with a lexicon-gated
# rule. See pipeline/translation/sinhala_normalize.py and docs/sinhala_zwj_repair.md.
ZWJ_FIX = os.environ.get("UC_ZWJ_FIX", "1") not in ("0", "false", "False")
ZWJ_CORPUS = Path(
    os.environ.get("UC_ZWJ_CORPUS", ROOT_DIR / "CPT-Dataset.txt")).resolve()
ZWJ_VOCAB = Path(
    os.environ.get("UC_ZWJ_VOCAB",
                   ROOT_DIR / "tokenizer" / "unigram_32000_0.9995.vocab")).resolve()
ZWJ_LEXICON_CACHE = Path(
    os.environ.get("UC_ZWJ_LEXICON_CACHE", DATA_DIR / "sinhala_lexicon.pkl")).resolve()
ZWJ_MIN_FREQ = int(os.environ.get("UC_ZWJ_MIN_FREQ", "5"))

# Where translated splits are written (one resumable JSONL per split).
OUTPUT_DIR = Path(
    os.environ.get("UC_OUTPUT_DIR", ROOT_DIR / "data" / "translated")
).resolve()

# Exact source-character totals per split (from stage 2), used to project full
# run time in the benchmark when the live stats JSON is not on disk.
SPLIT_TOTAL_CHARS = {
    "train_sft": 1_181_936_140,
    "test_sft": 130_616_688,
    "train_gen": 1_088_276_995,
    "test_gen": 119_820_387,
}

# Dialogue counts per split (from the dataset card / stage 1), used to estimate
# how much of a split is already done when resuming a translation run.
SPLIT_NUM_DIALOGUES = {
    "train_sft": 207_865,
    "test_sft": 23_110,
    "train_gen": 256_032,
    "test_gen": 28_304,
}


def group_total_chars(group: str) -> int:
    """Total source characters for a split group ('sft' / 'gen')."""
    return sum(SPLIT_TOTAL_CHARS.get(s, 0) for s in SPLIT_GROUPS.get(group, []))

# ---------------------------------------------------------------------------
# Split discovery
# ---------------------------------------------------------------------------
_SHARD_RE = re.compile(r"^(?P<split>.+?)-\d{5}-of-\d{5}.*\.parquet$")


def discover_splits(data_dir: Path | None = None) -> dict[str, list[Path]]:
    """Return {split_name: [sorted shard paths]} for every parquet file found.

    Filenames follow the HuggingFace convention
    ``<split>-00000-of-00001-<hash>.parquet``; the split name is the portion
    before the ``-NNNNN-of-NNNNN`` shard marker. Files that do not match the
    convention are grouped under their stem so nothing is silently dropped.
    """
    data_dir = Path(data_dir or DATA_DIR)
    splits: dict[str, list[Path]] = {}
    if not data_dir.exists():
        return splits
    for path in sorted(data_dir.rglob("*.parquet")):
        m = _SHARD_RE.match(path.name)
        split = m.group("split") if m else path.stem
        splits.setdefault(split, []).append(path)
    for split in splits:
        splits[split].sort()
    return splits


def ordered_splits(data_dir: Path | None = None) -> list[str]:
    """Discovered split names, EXPECTED_SPLITS first (in canonical order)."""
    found = discover_splits(data_dir)
    ordered = [s for s in EXPECTED_SPLITS if s in found]
    ordered += [s for s in sorted(found) if s not in EXPECTED_SPLITS]
    return ordered


if __name__ == "__main__":
    print(f"ROOT_DIR      = {ROOT_DIR}")
    print(f"DATA_DIR      = {DATA_DIR}")
    print(f"RESULTS_DIR   = {RESULTS_DIR}")
    print(f"TOKENIZER_DIR = {TOKENIZER_DIR} (exists={TOKENIZER_DIR.exists()})")
    print(f"HF_REPO       = {HF_REPO}")
    print("Discovered splits:")
    for split, shards in discover_splits().items():
        total = sum(p.stat().st_size for p in shards)
        print(f"  {split:<12} {len(shards)} shard(s)  {total/1e6:8.1f} MB")
