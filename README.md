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
| `08_tokenization_report.py` | `TOKENIZATION_REPORT.md` | Markdown render of stage 7 (no data pass) |
| `09_translate.py` | `data/translated/<split>.sinhala.jsonl` | **English→Sinhala translation** with NLLB-200-3.3B (resumable) |

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

## Stage 9 — Translation (English→Sinhala, NLLB-200)

`09_translate.py` is the actual translation stage (standalone; needs
`transformers`/`torch`/`sentencepiece`). It streams the parquet shards and, per
message: masks code/URLs/math/markup (`mt_preprocess`, reusing the stage-3/5
`detectors`), sentence-segments to fit NLLB's 512-token limit, batch-translates
`eng_Latn → sin_Sinh` with NLLB-200-3.3B, then restores masks and reassembles
the dialogue with `prompt_id`/roles/newlines preserved. Output is one **resumable**
JSONL per split under `UC_OUTPUT_DIR` (default `data/translated/`); re-running
skips `prompt_id`s already written.

```bash
pip install "transformers>=4.40" "torch>=2.1" sentencepiece sacremoses nltk

# Project run time for assorted throughputs (no model load — pure arithmetic):
python 09_translate.py estimate --chars-per-sec 150 300 600 1000

# Measure real throughput on THIS machine and project full SFT/GEN run time:
python 09_translate.py bench --split test_sft --n 50
UC_DEVICE=cpu UC_CPU_THREADS=4 python 09_translate.py bench --n 50   # mimic the 4-vCPU VM

# Translate the locally-present test splits (resumable):
python 09_translate.py run --splits test_sft test_gen

# End-to-end: download missing splits, then download the model, then translate.
python 09_translate.py run --download --splits train_sft test_sft   # SFT only
python 09_translate.py run --download                               # whole dataset
```

With `--download`, stage 9 first pulls any missing requested splits via the same
resumable `snapshot_download` stage 0 uses (parquet — nothing to unpack), then
loads NLLB and translates. Bare `--download` (no `--splits`) targets the whole
dataset. Throughout the run it logs a live **job line** — elapsed, src-char/s,
% done, ETA and projected **total time** — e.g.:

```
translate:   test_sft: 1,280/23,110 dialogues | elapsed 9m 12s | 263 src-char/s | 5.41% done | ETA 5.8 days | est. total 5.9 days
```

Knobs (env, see `config.py`): `UC_NLLB_MODEL`, `UC_DEVICE` (auto|cuda|cpu),
`UC_DTYPE` (auto|fp32|fp16|bf16), `UC_CPU_THREADS`, `UC_TRANSLATE_BATCH`,
`UC_NUM_BEAMS`, `UC_OUTPUT_DIR`, `UC_MASK`.

**Timing note (4-vCPU CPU VM).** NLLB-3.3B is memory-bound and slow on CPU.
Realistic throughput is ~150–400 src-char/s with vanilla transformers (bf16,
AMX) and ~500–1200 src-char/s with the optional CTranslate2 int8 path. That puts
the **full SFT** split (1.31 B chars) at **~3 weeks (int8) to ~2 months
(vanilla)** and **GEN** (1.21 B chars) at roughly the same; even the local
`test_*` slices are ~2–6 days each. Use this VM to validate the pipeline on a
few hundred dialogues (`--limit`), and run the full corpus on a GPU.

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
