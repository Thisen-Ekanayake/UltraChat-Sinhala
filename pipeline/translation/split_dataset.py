#!/usr/bin/env python3
"""Split the SFT corpus into N balanced parts for job-per-part translation.

With a large GPU (e.g. an MI300X with 192 GB) it is convenient to translate the
SFT split as several independent, resumable jobs running in parallel. This
module streams the source parquet shards and round-robins every dialogue into N
parts (``part_01.parquet`` … ``part_NN.parquet``), which keeps the parts close
to equal in both dialogue count and character volume. ``prompt_id`` is preserved
on every record, so the original train/test membership is always recoverable by
joining back to the source dataset — the parts mix train_sft and test_sft only
to balance size.

It also writes ``parts_manifest.json`` (per-part character and dialogue counts)
which the translation stage reads to show accurate per-part progress and ETA.

Usage (run from the repo root):
    python -m pipeline.translation.split_dataset                 # SFT -> 10 parts
    python -m pipeline.translation.split_dataset --parts 8 --out data/parts
    python -m pipeline.translation.split_dataset --splits train_sft test_sft
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline import config
from pipeline.common import fmt_int, get_logger, iter_messages

log = get_logger("split")

MANIFEST_NAME = "parts_manifest.json"
_FLUSH_ROWS = 2000          # rows buffered per part before a parquet write


def _record_chars(row: dict) -> int:
    return sum(len(m["content"]) for m in iter_messages(row))


def split_dataset(splits: list[str], parts: int, out_dir: Path,
                  name_prefix: str = "part") -> dict:
    discovered = config.discover_splits()
    sources: list[Path] = []
    for s in splits:
        if s not in discovered:
            log.warning("Split %s not found under %s; skipping.", s, config.DATA_DIR)
            continue
        sources.extend(discovered[s])
    if not sources:
        log.error("No source shards found for splits %s under %s. "
                  "Run `python -m pipeline.analysis.download_dataset` first.",
                  splits, config.DATA_DIR)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pq.ParquetFile(sources[0]).schema_arrow
    # Part names carry a dataset prefix (e.g. part_NN for SFT, gen_NN for GEN) so
    # the parts, outputs, logs and per-prefix concurrency throttle never collide.
    names = [f"{name_prefix}_{i + 1:02d}" for i in range(parts)]
    writers = [pq.ParquetWriter(out_dir / f"{n}.parquet", schema) for n in names]
    buffers: list[list[dict]] = [[] for _ in range(parts)]
    counts = [0] * parts
    chars = [0] * parts

    def flush(p: int) -> None:
        if buffers[p]:
            writers[p].write_table(pa.Table.from_pylist(buffers[p], schema=schema))
            buffers[p].clear()

    log.info("Splitting %s (%d shard(s)) into %d parts -> %s",
             splits, len(sources), parts, out_dir)
    idx = 0
    for shard in sources:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=config.BATCH_SIZE):
            for row in batch.to_pylist():
                p = idx % parts                       # round-robin = balanced
                buffers[p].append(row)
                counts[p] += 1
                chars[p] += _record_chars(row)
                idx += 1
                if len(buffers[p]) >= _FLUSH_ROWS:
                    flush(p)
    for p in range(parts):
        flush(p)
        writers[p].close()

    manifest = {
        "source_splits": splits,
        "parts": parts,
        "total_dialogues": sum(counts),
        "total_chars": sum(chars),
        "per_part": {names[p]: {"dialogues": counts[p], "chars": chars[p]}
                     for p in range(parts)},
    }
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("Wrote %d parts (%s dialogues, %s chars). Manifest: %s",
             parts, fmt_int(sum(counts)), fmt_int(sum(chars)),
             out_dir / MANIFEST_NAME)
    for p in range(parts):
        log.info("  %s: %s dialogues, %s chars",
                 names[p], fmt_int(counts[p]), fmt_int(chars[p]))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=list(config.SFT_SPLITS),
                    help="source splits to pool and split (default: SFT).")
    ap.add_argument("--parts", type=int, default=10, help="number of parts (default 10).")
    ap.add_argument("--out", default=str(config.DATA_DIR / "parts"),
                    help="output directory for <prefix>_NN.parquet (default: data/parts).")
    ap.add_argument("--name-prefix", default="part",
                    help="part filename prefix, e.g. 'part' (SFT) or 'gen' (GEN).")
    args = ap.parse_args()
    split_dataset(args.splits, args.parts, Path(args.out).resolve(), args.name_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
