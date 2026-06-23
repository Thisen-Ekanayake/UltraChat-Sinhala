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
# Scripts, data/ and results/ all live under this directory (the repo root).
ROOT_DIR = Path(__file__).resolve().parent

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

# The two splits relevant to the downstream Sinhala SFT translation effort.
SFT_SPLITS = ["train_sft", "test_sft"]

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
    print(f"ROOT_DIR    = {ROOT_DIR}")
    print(f"DATA_DIR    = {DATA_DIR}")
    print(f"RESULTS_DIR = {RESULTS_DIR}")
    print(f"HF_REPO     = {HF_REPO}")
    print("Discovered splits:")
    for split, shards in discover_splits().items():
        total = sum(p.stat().st_size for p in shards)
        print(f"  {split:<12} {len(shards)} shard(s)  {total/1e6:8.1f} MB")
