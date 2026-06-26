#!/usr/bin/env python3
"""Convert finalized .sinhala.jsonl splits to compressed parquet, reusing the
original ultrachat_200k schema (messages as list<struct<content, role>>, HF
metadata preserved). Streams in row-group chunks so memory stays bounded on a
small VM, and gives a ~4x size reduction vs jsonl.

Usage:
  python3 tools/jsonl_to_parquet.py <schema_parquet> <base> <suffix> <split> [split ...]
  # e.g. schema from any original ultrachat_200k parquet:
  python3 tools/jsonl_to_parquet.py data/src/part_01.parquet data/final_split \
      .sinhala.jsonl train_sft test_sft train_gen test_gen

Compression defaults to zstd; override with UC_PARQUET_COMPRESSION (snappy/gzip/...).
"""
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_FROM, BASE, SUFFIX = sys.argv[1], sys.argv[2], sys.argv[3]
SPLITS = sys.argv[4:]
CHUNK = 20000
COMP = os.environ.get("UC_PARQUET_COMPRESSION", "zstd")

schema = pq.read_schema(SCHEMA_FROM)
print("schema:", schema)

for s in SPLITS:
    inj = os.path.join(BASE, s + SUFFIX)
    outp = os.path.join(BASE, s + ".parquet")
    writer = pq.ParquetWriter(outp, schema, compression=COMP)
    batch, n = [], 0

    def flush():
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            batch.clear()

    for line in open(inj, encoding="utf-8"):
        if not line.strip():
            continue
        batch.append(json.loads(line))
        n += 1
        if len(batch) >= CHUNK:
            flush()
    flush()
    writer.close()
    js, pq_sz = os.path.getsize(inj), os.path.getsize(outp)
    print(f"{s}: {n:,} rows | jsonl {js/1e6:,.0f} MB -> parquet {pq_sz/1e6:,.0f} MB "
          f"({100*pq_sz/js:.0f}% of jsonl, {COMP})", flush=True)
