# UltraChat 200k — English→Sinhala Analysis & Translation Pipeline

Modular, streaming pipeline over the `HuggingFaceH4/ultrachat_200k` corpus: it
first **characterises** the data (structure, volume, MT-risk register, cost
model, SinLlama token counts) and then **translates** it English→Sinhala with
NLLB-200. Organised as the importable `pipeline` package; designed to run on a
GCP VM.

## Repository layout

All Python lives in the importable `pipeline` package; stages run as modules
(`python -m pipeline.<group>.<stage>`) from the repo root. Shell wrappers live in
`scripts/`.

```
pipeline/
  config.py          # central config (paths, NLLB knobs, cost model)
  common.py          # streaming readers, accumulators, JSON I/O, timers
  detectors.py       # shared MT-risk regexes (analysis + translation)
  analysis/          # stages 0–6: download + corpus analysis + report
  tokenization/      # stages 7–8: SinLlama token counts + report
  translation/       # NLLB en→si translation + masking/segmentation
scripts/
  run_analysis.sh    # orchestrates analysis stages 0–6
  run_translate.sh   # wrapper for the translation stage
data/                # parquet shards (+ data/translated/ output)
results/             # JSON artifacts + generated reports
```

## Quick start (analysis)

```bash
bash scripts/run_analysis.sh                       # install deps → download → analyse → report
# subset / reuse options:
bash scripts/run_analysis.sh --splits "train_sft test_sft"   # download SFT splits only
bash scripts/run_analysis.sh --skip-download                 # reuse parquet already in UC_DATA_DIR
bash scripts/run_analysis.sh --skip-install                  # deps already installed
```

Final report: `results/ANALYSIS_REPORT.md`. Intermediate JSON (one per stage)
and a timestamped run log are also written to `results/`.

## Stages (each independently runnable for debugging)

| Module | Output | What it measures |
|---|---|---|
| `pipeline.analysis.download_dataset` | `data/*.parquet` | Resumable HuggingFace snapshot download |
| `pipeline.analysis.analyze_structure` | `01_structure.json` | Turns, roles, alternation, schema integrity |
| `pipeline.analysis.analyze_text_stats` | `02_text_stats.json` | Char/token volume, length distributions, **cost model** |
| `pipeline.analysis.analyze_content_features` | `03_content_features.json` | Prevalence of code/markup/math/URLs |
| `pipeline.analysis.analyze_unicode_scripts` | `04_unicode_scripts.json` | Non-ASCII & per-script (multilingual) composition |
| `pipeline.analysis.analyze_translation_risks` | `05_translation_risks.json` | Sinhala MT **risk register** + examples |
| `pipeline.analysis.aggregate_report` | `ANALYSIS_REPORT.md` | Academic markdown report (no data pass) |
| `pipeline.tokenization.tokenize_splits` | `07_tokenization.json` | SinLlama token counts/distributions, **sft & gen separately** |
| `pipeline.tokenization.tokenization_report` | `TOKENIZATION_REPORT.md` | Markdown render of stage 7 (no data pass) |
| `pipeline.translation.translate` | `data/translated/<split>.sinhala.jsonl` | **English→Sinhala translation** with NLLB-200-3.3B (resumable) |

Run a single stage directly, e.g. `python -m pipeline.analysis.analyze_content_features`.

`scripts/run_analysis.sh` runs stages 0–6. **Tokenization is standalone** (not in
`run_analysis.sh`) because it needs the SinLlama model directory, supplied
separately on the VM:

```bash
pip install tokenizers                    # lightweight; sufficient for counting
#   PEP 668 VM: pip install --user tokenizers  (or --break-system-packages)
UC_TOKENIZER_DIR=/path/to/SinLlama_merged_bf16 python -m pipeline.tokenization.tokenize_splits
```

Stage 7 loads the tokenizer from `tokenizer.json` via the lightweight
`tokenizers` package and needs no `transformers`/`torch`. If `transformers` is
present it is used (and gives identical counts); otherwise the `tokenizers`
fallback is used automatically.

## Translation (English→Sinhala, NLLB-200)

`pipeline.translation.translate` is the actual translation stage (needs
`transformers`/`torch`/`sentencepiece`). It streams the parquet shards and, per
message: masks code/URLs/math/markup (`pipeline.translation.mt_preprocess`,
reusing the analysis `detectors`), sentence-segments to fit NLLB's 512-token
limit, batch-translates `eng_Latn → sin_Sinh` with NLLB-200-3.3B, then restores
masks and reassembles the dialogue with `prompt_id`/roles/newlines preserved.
Output is one **resumable** JSONL per split under `UC_OUTPUT_DIR` (default
`data/translated/`); re-running skips `prompt_id`s already written.

```bash
pip install "transformers>=4.40" "torch>=2.1" sentencepiece sacremoses nltk

# Project run time for assorted throughputs (no model load — pure arithmetic):
bash scripts/run_translate.sh estimate --chars-per-sec 150 300 600 1000

# Measure real throughput on THIS machine and project full SFT/GEN run time:
bash scripts/run_translate.sh bench --split test_sft --n 50
UC_DEVICE=cpu UC_CPU_THREADS=4 bash scripts/run_translate.sh bench --n 50   # mimic the 4-vCPU VM

# Translate the locally-present test splits (resumable):
bash scripts/run_translate.sh run --splits test_sft test_gen

# End-to-end: download missing splits, then download the model, then translate.
bash scripts/run_translate.sh run --download --splits train_sft test_sft   # SFT only
bash scripts/run_translate.sh run --download                               # whole dataset
```

(Equivalently `python -m pipeline.translation.translate <subcommand> …`.)

With `--download`, the stage first pulls any missing requested splits via the
same resumable `snapshot_download` the download stage uses (parquet — nothing to
unpack), then loads NLLB and translates. Bare `--download` (no `--splits`)
targets the whole dataset. Throughout the run it logs a live **job line** —
elapsed, src-char/s, % done, ETA and projected **total time** — e.g.:

```
translate:   test_sft: 1,280/23,110 dialogues | elapsed 9m 12s | 263 src-char/s | 5.41% done | ETA 5.8 days | est. total 5.9 days
```

Knobs (env, see `pipeline/config.py`): `UC_NLLB_MODEL`, `UC_DEVICE`
(auto|cuda|cpu), `UC_DTYPE` (auto|fp32|fp16|bf16), `UC_CPU_THREADS`,
`UC_TRANSLATE_BATCH`, `UC_NUM_BEAMS`, `UC_OUTPUT_DIR`, `UC_MASK`.

Knobs that also apply: `UC_TRANSLATE_BATCH`, `UC_DIALOGUE_CHUNK`,
`UC_MAX_SEGMENT_CHARS`, `UC_MAX_INPUT_TOKENS`. The translator **guarantees**
NLLB's 512-token limit (over-long segments are split at token boundaries, never
truncated) and **length-buckets** segments before batching, which is the main
throughput lever on a GPU.

### Sharded GPU translation (AMD MI300X / ROCm)

For a big GPU, split the SFT corpus into N balanced parts and run a resumable
job per part (parts mix train+test to balance size; `prompt_id` is preserved so
train/test membership is recoverable):

```bash
# one-time setup on the pod (creates .venv with ROCm torch + deps)
ROCM_INDEX=https://download.pytorch.org/whl/rocm6.3 bash scripts/setup_pod_rocm.sh
source .venv/bin/activate

python -m pipeline.analysis.download_dataset --splits train_sft test_sft
python -m pipeline.translation.split_dataset --parts 10        # -> data/parts/part_01..10.parquet + manifest

# one part (writes data/translated_part_01/part_01.sinhala.jsonl, resumable):
bash scripts/run_translate_mi300x.sh part_01
# …or launch all parts in parallel (one tmux window each), capped concurrency:
PARTS=10 CONCURRENCY=4 bash scripts/run_all_parts_mi300x.sh
```

ROCm exposes the GPU through torch's CUDA API, so `UC_DEVICE=cuda` is correct;
the MI300X runs `bf16` natively. Each job's per-part wall-clock is the final
`DONE … in <time>` / `total job time` line in `results/translate_part_NN_*.log`.

**Timing note (4-vCPU CPU VM).** NLLB-3.3B is memory-bound and slow on CPU
(~150–400 src-char/s; full SFT ≈ weeks), which is why translation runs on the
MI300X. Use the CPU VM only to validate the pipeline on a few hundred dialogues
(`--limit`).

## Configuration (environment variables, see `pipeline/config.py`)

| Var | Default | Meaning |
|---|---|---|
| `UC_DATA_DIR` | `./data` | Directory of `*.parquet` shards |
| `UC_RESULTS_DIR` | `./results` | JSON artifacts + report |
| `UC_TOKENIZER_DIR` | `./SinLlama_merged_bf16` | SinLlama model dir for tokenization stage |
| `UC_HF_REPO` | `HuggingFaceH4/ultrachat_200k` | Source dataset |
| `UC_OUTPUT_DIR` | `./data/translated` | Translated JSONL output dir |

## Design notes

- **Streaming.** Every stage reads parquet via `pyarrow` `iter_batches`; peak
  memory is bounded by `BATCH_SIZE`, independent of the ~3 GB corpus. The cost
  is one disk pass per stage — a deliberate trade for modularity/debuggability.
- **Exact vs estimated.** Character counts are exact (the MT billing unit).
  Token counts use `tiktoken` (`cl100k_base`) if installed, else a `chars/4`
  heuristic; monetary figures are list-price estimates. All are labelled.
- **Shared detectors.** The analysis risk stages and the translation masker
  import the same compiled patterns from `pipeline/detectors.py`, so prevalence,
  the risk register, and what gets masked before translation are all consistent.
- **Validated** on the `test_sft` / `test_gen` splits (51,414 dialogues).
