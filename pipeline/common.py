"""Shared utilities for the UltraChat 200k analysis pipeline.

Every analysis script in this pipeline performs a single streaming pass over the
parquet shards. Streaming (never materialising a whole split in memory) keeps
peak RAM bounded by the batch size regardless of the ~3 GB dataset size, at the
cost of one disk read per script. This trade favours debuggability and
modularity, as requested, over a single fused pass.

Conventions
-----------
* A *record* is one dict with keys ``prompt`` (str), ``prompt_id`` (str) and
  ``messages`` (list of {"role", "content"}).
* A *message* is one {"role", "content"} dict.
* Each script writes exactly one JSON file into RESULTS_DIR; the aggregator
  reads them back. JSON is the single source of truth between stages.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from pipeline import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Streaming readers
# ---------------------------------------------------------------------------
def iter_split_records(
    split: str, shards: list[Path], batch_size: int = config.BATCH_SIZE
) -> Iterator[dict[str, Any]]:
    """Yield records from all shards of ``split`` without loading them at once."""
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def iter_messages(record: dict[str, Any]) -> Iterator[dict[str, str]]:
    """Yield well-formed messages from a record, tolerating schema noise."""
    msgs = record.get("messages") or []
    for m in msgs:
        if isinstance(m, dict):
            yield {"role": m.get("role", ""), "content": m.get("content") or ""}


# ---------------------------------------------------------------------------
# Streaming numeric summary
# ---------------------------------------------------------------------------
class LengthAccumulator:
    """Collects integer lengths and reports exact percentiles via numpy.

    Storing values (rather than a streaming sketch) gives exact quantiles. For
    the full dataset this is on the order of a few million int32 values
    (tens of MB), which is acceptable on a VM and keeps the statistics exact
    and reproducible.
    """

    def __init__(self) -> None:
        self._values: list[int] = []

    def add(self, value: int) -> None:
        self._values.append(value)

    def extend(self, values) -> None:
        self._values.extend(values)

    @property
    def n(self) -> int:
        return len(self._values)

    def summary(self) -> dict[str, float]:
        if not self._values:
            return {"count": 0, "sum": 0, "min": 0, "max": 0, "mean": 0.0,
                    "p50": 0, "p90": 0, "p95": 0, "p99": 0, "std": 0.0}
        arr = np.asarray(self._values, dtype=np.int64)
        p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
        return {
            "count": int(arr.size),
            "sum": int(arr.sum()),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(p50),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
        }


# ---------------------------------------------------------------------------
# JSON result I/O
# ---------------------------------------------------------------------------
def result_path(name: str) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.RESULTS_DIR / name


def save_result(name: str, data: dict[str, Any]) -> Path:
    path = result_path(name)
    payload = {"_generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **data}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def load_result(name: str) -> dict[str, Any]:
    with open(result_path(name), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def require_splits(logger: logging.Logger) -> dict[str, list[Path]]:
    """Discover splits, abort with a clear message if none are present."""
    splits = config.discover_splits()
    if not splits:
        logger.error(
            "No parquet shards found under %s. Run "
            "`python -m pipeline.analysis.download_dataset` first "
            "or set UC_DATA_DIR.", config.DATA_DIR,
        )
        sys.exit(2)
    return splits


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}"


class StepTimer:
    """Context manager that logs wall-clock duration of a processing step."""

    def __init__(self, logger: logging.Logger, label: str) -> None:
        self.logger, self.label = logger, label

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info("START %s", self.label)
        return self

    def __exit__(self, *exc):
        self.logger.info("DONE  %s (%.1fs)", self.label, time.time() - self.t0)
