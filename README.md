# UltraChat 200k — Pre-Translation Analysis Pipeline

Modular, streaming analysis of the `HuggingFaceH4/ultrachat_200k` corpus to
characterise it **before** an English→Sinhala machine-translation effort and to
produce a prioritised risk register and cost model. Designed to run on a GCP VM.

## Quick start

```bash
bash run_all.sh                       # install deps → download → analyse → report
# subset / reuse options:
bash run_all.sh --splits "train_sft test_sft"   # download SFT splits only
bash run_all.sh --skip-download                 # reuse parquet already in UC_DATA_DIR
bash run_all.sh --skip-install                  # deps already installed
```

Final report: `results/ANALYSIS_REPORT.md`. Intermediate JSON (one per stage)
and a timestamped run log are also written to `results/`.

## Stages (each independently runnable for debugging)

| Script | Output | What it measures |
|---|---|---|
| `00_download_dataset.py` | `data/*.parquet` | Resumable HuggingFace snapshot download |
| `01_analyze_structure.py` | `01_structure.json` | Turns, roles, alternation, schema integrity |
| `02_analyze_text_stats.py` | `02_text_stats.json` | Char/token volume, length distributions, **cost model** |
| `03_analyze_content_features.py` | `03_content_features.json` | Prevalence of code/markup/math/URLs |
| `04_analyze_unicode_scripts.py` | `04_unicode_scripts.json` | Non-ASCII & per-script (multilingual) composition |
| `05_analyze_translation_risks.py` | `05_translation_risks.json` | Sinhala MT **risk register** + examples |
| `06_aggregate_report.py` | `ANALYSIS_REPORT.md` | Academic markdown report (no data pass) |
| `07_tokenize_splits.py` | `07_tokenization.json` | SinLlama token counts/distributions, **sft & gen separately** |

Run a single stage directly, e.g. `python 03_analyze_content_features.py`.

`run_all.sh` runs stages 0–6. **Stage 7 is standalone** (not in `run_all.sh`)
because it needs the SinLlama model directory, supplied separately on the VM:

```bash
pip install tokenizers                    # lightweight; sufficient for counting
#   PEP 668 VM: pip install --user tokenizers  (or --break-system-packages)
UC_TOKENIZER_DIR=/path/to/SinLlama_merged_bf16 python 07_tokenize_splits.py
```

Stage 7 loads the tokenizer from `tokenizer.json` via the lightweight
`tokenizers` package and needs no `transformers`/`torch`. If `transformers` is
present it is used (and gives identical counts); otherwise the `tokenizers`
fallback is used automatically.

## Configuration (environment variables, see `config.py`)

| Var | Default | Meaning |
|---|---|---|
| `UC_DATA_DIR` | `./data` | Directory of `*.parquet` shards |
| `UC_RESULTS_DIR` | `./results` | JSON artifacts + report |
| `UC_TOKENIZER_DIR` | `./SinLlama_merged_bf16` | SinLlama model dir for stage 7 |
| `UC_HF_REPO` | `HuggingFaceH4/ultrachat_200k` | Source dataset |

## Design notes

- **Streaming.** Every stage reads parquet via `pyarrow` `iter_batches`; peak
  memory is bounded by `BATCH_SIZE`, independent of the ~3 GB corpus. The cost
  is one disk pass per stage — a deliberate trade for modularity/debuggability.
- **Exact vs estimated.** Character counts are exact (the MT billing unit).
  Token counts use `tiktoken` (`cl100k_base`) if installed, else a `chars/4`
  heuristic; monetary figures are list-price estimates. All are labelled.
- **Shared detectors.** Stages 3 and 5 import the same compiled patterns from
  `detectors.py`, so prevalence and the risk register are always consistent.
- **Validated** on the `test_sft` / `test_gen` splits (51,414 dialogues).
