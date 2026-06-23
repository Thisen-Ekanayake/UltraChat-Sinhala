#!/usr/bin/env python3
"""Stage 0 — download the UltraChat 200k parquet shards from HuggingFace.

Fetches only the parquet data files (and README) for the configured repo into
``UC_DATA_DIR``. Designed to run on a fresh GCP VM. Uses ``snapshot_download``
which is resumable: re-running skips files already present (hash-checked), so a
dropped connection is safe to retry.

Usage (run from the repo root)
-----
    python -m pipeline.analysis.download_dataset            # full dataset into UC_DATA_DIR
    python -m pipeline.analysis.download_dataset --splits train_sft test_sft
    python -m pipeline.analysis.download_dataset --list     # list remote files only

The downloaded shards are symlinked/copied into a flat layout under DATA_DIR so
that the analysis stages discover them via config.discover_splits().
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pipeline import config
from pipeline.common import fmt_int, get_logger

log = get_logger("download")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", nargs="*", default=None,
                   help="Restrict to these split prefixes (default: all).")
    p.add_argument("--list", action="store_true",
                   help="List remote parquet files and exit.")
    p.add_argument("--data-dir", default=str(config.DATA_DIR),
                   help="Target directory (default: UC_DATA_DIR).")
    return p.parse_args()


def list_remote_files() -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(config.HF_REPO, repo_type=config.HF_REPO_TYPE)
    return sorted(f for f in files if f.endswith(".parquet"))


def build_allow_patterns(splits: list[str] | None) -> list[str]:
    if not splits:
        return ["*.parquet", "README.md"]
    # HF stores files under data/<split>-*.parquet
    pats = []
    for s in splits:
        pats += [f"*{s}-*.parquet", f"{s}-*.parquet"]
    pats.append("README.md")
    return pats


def flatten_into_data_dir(snapshot_dir: Path, data_dir: Path) -> int:
    """Mirror every downloaded *.parquet into a flat DATA_DIR via hardlink/copy."""
    data_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(Path(snapshot_dir).rglob("*.parquet")):
        dst = data_dir / src.name
        if dst.exists():
            continue
        try:
            dst.hardlink_to(src.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(src.resolve(), dst)
        n += 1
    # carry the README along for provenance
    for readme in Path(snapshot_dir).rglob("README.md"):
        dst = data_dir / "DATASET_README.md"
        if not dst.exists():
            try:
                shutil.copy2(readme.resolve(), dst)
            except OSError:
                pass
        break
    return n


def main() -> int:
    args = _parse_args()

    if args.list:
        log.info("Remote parquet files in %s:", config.HF_REPO)
        for f in list_remote_files():
            print(" ", f)
        return 0

    from huggingface_hub import snapshot_download

    data_dir = Path(args.data_dir).resolve()
    allow = build_allow_patterns(args.splits)
    log.info("Repo:        %s (%s)", config.HF_REPO, config.HF_REPO_TYPE)
    log.info("Patterns:    %s", allow)
    log.info("Target dir:  %s", data_dir)

    snapshot_dir = snapshot_download(
        repo_id=config.HF_REPO,
        repo_type=config.HF_REPO_TYPE,
        allow_patterns=allow,
    )
    log.info("Snapshot cached at %s", snapshot_dir)

    n = flatten_into_data_dir(Path(snapshot_dir), data_dir)
    log.info("Linked %d parquet file(s) into %s", n, data_dir)

    # Summarise what is now available to the analysis stages.
    splits = config.discover_splits(data_dir)
    if not splits:
        log.error("No parquet shards present after download — check patterns.")
        return 2
    log.info("Available splits:")
    for split, shards in splits.items():
        size = sum(p.stat().st_size for p in shards)
        log.info("  %-12s %d shard(s)  %s bytes", split, len(shards), fmt_int(size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
