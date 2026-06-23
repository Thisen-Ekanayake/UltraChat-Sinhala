#!/usr/bin/env python3
"""Stage 9 — translate UltraChat 200k from English to Sinhala with NLLB-200.

Open-source NMT is the only budget-viable path at this corpus volume (report
§8.1). This stage streams the parquet shards, and for every message:

  1. masks code / URLs / math / markup so MT cannot corrupt them (mt_preprocess),
  2. sentence-segments it to fit NLLB's 512-token input limit,
  3. batch-translates the segments eng_Latn -> sin_Sinh with NLLB-200-3.3B,
  4. restores the masked spans and reassembles the message with its newline
     structure, role and order preserved.

Output is one resumable JSONL per split under ``UC_OUTPUT_DIR`` (mirroring the
input schema: ``prompt`` / ``prompt_id`` / ``messages[{role, content}]``).
Re-running skips dialogues whose ``prompt_id`` is already written, so a long run
is checkpointed and crash-safe.

Subcommands
-----------
  run       translate one or more splits to disk (resumable)
  bench     translate the first N dialogues, measure throughput on THIS machine,
            and project full SFT / GEN run time
  estimate  print the SFT / GEN run-time projection for given chars/sec rates
            (no model load — pure arithmetic)

Run from the repo root as a module (or via scripts/run_translate.sh).

Examples
--------
  # how long would the full SFT / GEN splits take at assorted CPU throughputs?
  python -m pipeline.translation.translate estimate --chars-per-sec 150 300 600 1000

  # measure real throughput on this box (downloads NLLB-3.3B on first use):
  python -m pipeline.translation.translate bench --split test_sft --n 50
  UC_DEVICE=cpu UC_CPU_THREADS=4 python -m pipeline.translation.translate bench --n 50

  # translate the locally-present test splits:
  python -m pipeline.translation.translate run --splits test_sft test_gen

Config knobs (env): UC_NLLB_MODEL, UC_DEVICE, UC_DTYPE, UC_CPU_THREADS,
UC_TRANSLATE_BATCH, UC_NUM_BEAMS, UC_OUTPUT_DIR, UC_MASK — see config.py.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from pipeline import config
from pipeline.translation.mt_preprocess import from_segments, protect, restore, to_segments
from pipeline.common import (
    StepTimer, fmt_int, get_logger, iter_messages, iter_split_records,
    require_splits,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

log = get_logger("translate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def human_time(seconds: float) -> str:
    """Compact human duration: '45s', '12m 30s', '3h 20m', '6.4 days'."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    return f"{seconds / 86400:.1f} days"


def out_path(split: str):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return config.OUTPUT_DIR / f"{split}.sinhala.jsonl"


def load_done(path) -> set[str]:
    """prompt_ids already translated in a previous (possibly interrupted) run."""
    done: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["prompt_id"])
                except Exception:
                    continue
    return done


def record_src_chars(rec) -> int:
    return sum(len(m["content"]) for m in iter_messages(rec))


# Per-part totals: the named SFT/GEN splits have built-in constants; arbitrary
# parts (part_NN) get their totals from parts_manifest.json in the data dir.
def _load_part_manifest() -> dict:
    try:
        data = json.loads((config.DATA_DIR / "parts_manifest.json")
                          .read_text(encoding="utf-8"))
        return data.get("per_part", {})
    except Exception:
        return {}


_PART_MANIFEST = _load_part_manifest()


def split_chars(split: str) -> int:
    return (config.SPLIT_TOTAL_CHARS.get(split)
            or _PART_MANIFEST.get(split, {}).get("chars", 0))


def split_dialogues(split: str) -> int:
    return (config.SPLIT_NUM_DIALOGUES.get(split)
            or _PART_MANIFEST.get(split, {}).get("dialogues", 0))


# ---------------------------------------------------------------------------
# Dataset acquisition (reuses stage 0's HuggingFace download logic)
# ---------------------------------------------------------------------------
def ensure_splits(requested: list[str], present: dict, do_download: bool) -> dict:
    """Make sure every requested split's parquet is on disk.

    The UltraChat splits are HuggingFace parquet shards (the data itself — no
    archive to unpack). Missing ones are pulled with the same resumable
    ``snapshot_download`` stage 0 uses, then flattened into ``DATA_DIR``.
    Returns the re-discovered split map.
    """
    missing = [s for s in requested if s not in present]
    if not missing:
        return present
    if not do_download:
        log.warning("Requested splits not present and --download not set: %s",
                    ", ".join(missing))
        return present

    dl = importlib.import_module("pipeline.analysis.download_dataset")
    from huggingface_hub import snapshot_download

    allow = dl.build_allow_patterns(missing)
    with StepTimer(log, f"download splits {missing} from {config.HF_REPO}"):
        snap = snapshot_download(repo_id=config.HF_REPO,
                                 repo_type=config.HF_REPO_TYPE,
                                 allow_patterns=allow)
        n = dl.flatten_into_data_dir(Path(snap), config.DATA_DIR)
    log.info("Linked %d new parquet file(s) into %s", n, config.DATA_DIR)
    return config.discover_splits()


# ---------------------------------------------------------------------------
# Job-level progress / total-time tracking
# ---------------------------------------------------------------------------
class JobProgress:
    """Tracks elapsed time, throughput and ETA across all translated splits.

    ``total_chars`` is the source-character budget for the whole job (known
    constants); ``run_chars`` is what this process has translated so far.
    Resumed work is credited via :meth:`add_resumed` so the % / ETA reflect the
    real remaining work, not just this run.
    """

    def __init__(self, targets: list[str]) -> None:
        self.t0 = time.time()
        self.total_chars = sum(split_chars(s) for s in targets)
        self.resumed_chars = 0.0
        self.run_chars = 0

    def add_resumed(self, split: str, done_count: int) -> None:
        nd = split_dialogues(split)
        tc = split_chars(split)
        if nd and done_count:
            self.resumed_chars += min(done_count / nd, 1.0) * tc

    def add(self, chars: int) -> None:
        self.run_chars += chars

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    @property
    def rate(self) -> float:
        e = self.elapsed
        return self.run_chars / e if e > 0 and self.run_chars else 0.0

    def line(self) -> str:
        rate = self.rate
        if rate <= 0:
            return f"elapsed {human_time(self.elapsed)} | warming up…"
        done = self.resumed_chars + self.run_chars
        pct = 100 * done / self.total_chars if self.total_chars else 0.0
        remaining = max(self.total_chars - done, 0)
        return (f"elapsed {human_time(self.elapsed)} | {rate:.0f} src-char/s | "
                f"{pct:.2f}% done | ETA {human_time(remaining / rate)} | "
                f"est. total {human_time(self.total_chars / rate)}")


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------
class NllbTranslator:
    """Thin batched wrapper over an NLLB-200 seq2seq model."""

    def __init__(self) -> None:
        import torch
        import transformers
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # Quiet the repeated per-batch generation warnings; we configure
        # generation explicitly below so they carry no information.
        transformers.logging.set_verbosity_error()

        self.torch = torch
        device, dtype, dtype_name = self._resolve(torch)
        self.device = device

        if device == "cpu" and config.CPU_THREADS:
            torch.set_num_threads(config.CPU_THREADS)

        with StepTimer(log, f"load {config.NLLB_MODEL} -> {device}/{dtype_name}"):
            self.tok = AutoTokenizer.from_pretrained(
                config.NLLB_MODEL, src_lang=config.SRC_LANG)
            # transformers >=5 renamed torch_dtype -> dtype; support both.
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.NLLB_MODEL, dtype=dtype)
            except TypeError:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.NLLB_MODEL, torch_dtype=dtype)
            self.model.to(device).eval()
        self.bos = self.tok.convert_tokens_to_ids(config.TGT_LANG)

        # Bake generation settings into the model's generation_config so every
        # generate() call is parameter-free. This also removes the model's
        # default max_length (=200), which otherwise conflicts with
        # max_new_tokens and prints a warning on every batch.
        gc = self.model.generation_config
        gc.forced_bos_token_id = self.bos
        gc.max_new_tokens = config.MAX_NEW_TOKENS
        gc.max_length = None
        gc.num_beams = config.NUM_BEAMS

        # NLLB's positional limit is 512 tokens. Segments are char-bounded
        # upstream, but we additionally *guarantee* the limit here (no silent
        # truncation): any segment over the budget is split at token boundaries.
        self.max_in = config.MAX_INPUT_TOKENS
        # A segment shorter than this many chars cannot exceed max_in tokens
        # (assumes >= 2 chars/token), so the length check is skipped for it.
        self._safe_chars = self.max_in * 2

        threads = torch.get_num_threads() if device == "cpu" else None
        log.info("Translator ready: %s -> %s, batch=%d, beams=%d, max_in=%d%s",
                 config.SRC_LANG, config.TGT_LANG, config.TRANSLATE_BATCH,
                 config.NUM_BEAMS, self.max_in,
                 f", cpu_threads={threads}" if threads else "")

    @staticmethod
    def _resolve(torch):
        dtypes = {"fp32": torch.float32, "fp16": torch.float16,
                  "bf16": torch.bfloat16}
        device = config.TRANSLATE_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        name = config.TRANSLATE_DTYPE
        if name == "auto":
            name = "fp16" if device == "cuda" else "bf16"
        return device, dtypes[name], name

    def _fit_tokens(self, text: str) -> list[str]:
        """Split ``text`` so each piece is <= max_in tokens (NLLB's 512 limit).

        Cheap by design: a text below ``_safe_chars`` cannot exceed the budget,
        so it is returned untouched without tokenising. Only genuinely long
        segments are tokenised and cut at token boundaries (via offset mapping,
        with a proportional char-split fallback for slow tokenisers).
        """
        if len(text) <= self._safe_chars:
            return [text]
        try:
            enc = self.tok(text, add_special_tokens=True,
                           return_offsets_mapping=True)
            ids = enc["input_ids"]
            if len(ids) <= self.max_in:
                return [text]
            budget = self.max_in - 2          # reserve src-lang + eos per piece
            spans = [sp for sp in enc["offset_mapping"] if sp[1] > sp[0]]
            pieces, start, cnt = [], spans[0][0], 0
            for _s, e in spans:
                cnt += 1
                if cnt >= budget:
                    pieces.append(text[start:e])
                    start, cnt = e, 0
            if start < len(text):
                pieces.append(text[start:])
            return [p for p in pieces if p.strip()] or [text]
        except Exception:                     # slow tokenizer / no offsets
            approx = self._safe_chars
            return [text[i:i + approx] for i in range(0, len(text), approx)]

    def translate_many(self, texts: list[str], desc: str | None = None) -> list[str]:
        if not texts:
            return []
        # 1. Enforce the 512-token limit: expand any over-long segment into
        #    sub-pieces, remembering which original text each piece belongs to.
        pieces: list[str] = []
        parent: list[int] = []
        for i, t in enumerate(texts):
            for sub in self._fit_tokens(t):
                pieces.append(sub)
                parent.append(i)

        # 2. Length-bucketed batching: sort by length so each batch pads to a
        #    similar size. On a big GPU this is the dominant throughput win —
        #    padding short and long segments together wastes most of the compute.
        order = sorted(range(len(pieces)), key=lambda k: len(pieces[k]))
        out: list[str | None] = [None] * len(pieces)
        bs = config.TRANSLATE_BATCH
        bar = _tqdm(total=len(pieces), unit="seg", desc=desc, leave=False) \
            if desc and _tqdm else None
        with self.torch.inference_mode():
            for b in range(0, len(order), bs):
                idxs = order[b:b + bs]
                batch = [pieces[k] for k in idxs]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_in)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                gen = self.model.generate(**enc)
                dec = self.tok.batch_decode(gen, skip_special_tokens=True)
                for k, d in zip(idxs, dec):
                    out[k] = d
                if bar:
                    bar.update(len(idxs))
        if bar:
            bar.close()

        # 3. Reassemble: stitch sub-pieces back per original text, in order.
        results: list[list[str]] = [[] for _ in texts]
        for p_idx, par in enumerate(parent):
            results[par].append(out[p_idx] or "")
        return [" ".join(r) for r in results]

    def warmup(self) -> None:
        """First call pays lazy GPU/oneDNN init; do it off the clock."""
        self.translate_many(["Hello.", "This is a warmup sentence."])


# ---------------------------------------------------------------------------
# Per-dialogue preprocessing + reassembly
# ---------------------------------------------------------------------------
def _prep_text(text: str, mask: bool) -> dict:
    originals: list[str] = []
    masked = text
    if mask:
        masked, originals = protect(text)
    segments, layout = to_segments(masked)
    return {"originals": originals, "segments": segments, "layout": layout}


def _preprocess_record(rec: dict, mask: bool) -> dict:
    msgs = list(iter_messages(rec))
    prepped = [(m["role"], _prep_text(m["content"], mask)) for m in msgs]
    prompt = rec.get("prompt") or ""
    # The prompt usually duplicates messages[0]; only translate it separately
    # when it differs, otherwise reuse the translated first message.
    prompt_prep = None
    if prompt and not (msgs and prompt == msgs[0]["content"]):
        prompt_prep = _prep_text(prompt, mask)
    return {
        "prompt_id": rec.get("prompt_id", ""),
        "prompt": prompt,
        "prompt_dupes_msg0": bool(msgs and prompt == msgs[0]["content"]),
        "msgs": prepped,
        "prompt_prep": prompt_prep,
    }


def translate_records(records: list[dict], tr: NllbTranslator, mask: bool,
                      desc: str | None = None) -> list[dict]:
    """Translate a chunk of dialogues, batching every segment across them."""
    prepped = [_preprocess_record(r, mask) for r in records]

    # Flatten all segments (message slots, then the optional standalone prompt).
    flat: list[str] = []
    for pr in prepped:
        for _role, p in pr["msgs"]:
            flat.extend(p["segments"])
        if pr["prompt_prep"]:
            flat.extend(pr["prompt_prep"]["segments"])

    translated = tr.translate_many(flat, desc=desc)

    # Scatter the translations back in the exact order they were flattened.
    out: list[dict] = []
    idx = 0
    for pr in prepped:
        msgs_out = []
        for role, p in pr["msgs"]:
            n = len(p["segments"])
            content = from_segments(translated[idx:idx + n], p["layout"])
            content = restore(content, p["originals"])
            msgs_out.append({"role": role, "content": content})
            idx += n
        if pr["prompt_prep"]:
            p = pr["prompt_prep"]
            n = len(p["segments"])
            prompt_out = restore(from_segments(translated[idx:idx + n], p["layout"]),
                                 p["originals"])
            idx += n
        elif pr["prompt_dupes_msg0"] and msgs_out:
            prompt_out = msgs_out[0]["content"]
        else:
            prompt_out = pr["prompt"]
        out.append({"prompt": prompt_out, "prompt_id": pr["prompt_id"],
                    "messages": msgs_out})
    return out


# ---------------------------------------------------------------------------
# run: translate a split to disk (resumable)
# ---------------------------------------------------------------------------
def run_split(split, shards, tr, mask, job: JobProgress, limit=None) -> None:
    path = out_path(split)
    done = load_done(path)
    job.add_resumed(split, len(done))
    log.info("Split %s -> %s (%s already done)", split, path.name, fmt_int(len(done)))

    buf: list[dict] = []
    written = chars = 0
    t0 = time.time()

    def flush():
        nonlocal written, chars
        if not buf:
            return
        outs = translate_records(buf, tr, mask, desc=f"{split} segs")
        flush_chars = 0
        with open(path, "a", encoding="utf-8") as fh:
            for src, dst in zip(buf, outs):
                fh.write(json.dumps(dst, ensure_ascii=False) + "\n")
                flush_chars += record_src_chars(src)
        chars += flush_chars
        written += len(buf)
        job.add(flush_chars)
        log.info("  %s: %s/%s dialogues | %s", split, fmt_int(written),
                 fmt_int(split_dialogues(split) or written), job.line())
        buf.clear()

    for rec in iter_split_records(split, shards):
        if rec.get("prompt_id") in done:
            continue
        buf.append(rec)
        if len(buf) >= config.DIALOGUE_CHUNK:
            flush()
        if limit and written + len(buf) >= limit:
            break
    flush()
    log.info("DONE %s: %s new dialogues, %s src-chars in %s",
             split, fmt_int(written), fmt_int(chars),
             human_time(time.time() - t0))


# ---------------------------------------------------------------------------
# Projection / benchmark
# ---------------------------------------------------------------------------
def projection_table(rates: list[float]) -> str:
    scopes = [
        ("test_sft (local)", config.SPLIT_TOTAL_CHARS["test_sft"]),
        ("test_gen (local)", config.SPLIT_TOTAL_CHARS["test_gen"]),
        ("SFT  (train+test)", config.group_total_chars("sft")),
        ("GEN  (train+test)", config.group_total_chars("gen")),
        ("Whole dataset",     sum(config.SPLIT_TOTAL_CHARS.values())),
    ]
    head = ["scope", "chars"] + [f"@{int(r)} c/s" for r in rates]
    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join("---" for _ in head) + " |"]
    for name, chars in scopes:
        row = [name, fmt_int(chars)] + [human_time(chars / r) for r in rates]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def bench(split, shards, tr, mask, n) -> None:
    records = []
    for rec in iter_split_records(split, shards):
        records.append(rec)
        if len(records) >= n:
            break
    if not records:
        log.error("No records in split %s", split)
        return
    src_chars = sum(record_src_chars(r) for r in records)
    n_msgs = sum(1 for r in records for _ in iter_messages(r))

    tr.warmup()
    t0 = time.time()
    translate_records(records, tr, mask)
    dt = time.time() - t0

    cps = src_chars / dt
    log.info("BENCH on %s: %d dialogues / %d messages / %s src-chars in %s",
             split, len(records), n_msgs, fmt_int(src_chars), human_time(dt))
    log.info("  throughput: %.1f src-char/s | %.2f dialogues/s | %.1f messages/s",
             cps, len(records) / dt, n_msgs / dt)
    print("\nProjected wall-clock at the measured rate "
          f"({cps:.0f} src-char/s):\n")
    print(projection_table([cps]))
    print("\nNote: measured on THIS host. For the 4-vCPU CPU VM, re-run with "
          "UC_DEVICE=cpu UC_CPU_THREADS=4.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="translate splits to disk (resumable)")
    pr.add_argument("--splits", nargs="*", default=None)
    pr.add_argument("--download", action="store_true",
                    help="download any missing requested splits first "
                         "(default targets = whole dataset when set)")
    pr.add_argument("--limit", type=int, default=None,
                    help="cap dialogues per split (debug)")
    pr.add_argument("--no-mask", action="store_true")

    pb = sub.add_parser("bench", help="measure throughput + project full run")
    pb.add_argument("--split", default="test_sft")
    pb.add_argument("--n", type=int, default=50)
    pb.add_argument("--no-mask", action="store_true")

    pe = sub.add_parser("estimate", help="project run time for given chars/sec")
    pe.add_argument("--chars-per-sec", nargs="+", type=float,
                    default=[150, 300, 600, 1000])

    args = ap.parse_args()

    if args.cmd == "estimate":
        print(f"NLLB en->si projected wall-clock ({config.NLLB_MODEL}):\n")
        print(projection_table(args.chars_per_sec))
        return 0

    if args.cmd == "bench":
        splits = require_splits(log)
        if args.split not in splits:
            log.error("Split %s not present locally (have: %s)",
                      args.split, ", ".join(splits))
            return 2
        tr = NllbTranslator()
        mask = not args.no_mask and config.MASK_BEFORE_TRANSLATE
        bench(args.split, splits[args.split], tr, mask, args.n)
        return 0

    # cmd == "run": acquire data (optionally downloading) BEFORE loading the model.
    targets = args.splits or (
        list(config.EXPECTED_SPLITS) if args.download else config.ordered_splits())
    if not targets:
        log.error("Nothing to translate. Use --download, pass --splits, or run "
                  "`python -m pipeline.analysis.download_dataset` first.")
        return 2

    present = config.discover_splits()
    if args.download:
        present = ensure_splits(targets, present, do_download=True)
    available = [s for s in targets if s in present]
    for s in targets:
        if s not in present:
            log.warning("Split %s unavailable; skipping.", s)
    if not available:
        log.error("None of the requested splits are available under %s.",
                  config.DATA_DIR)
        return 2

    mask = not args.no_mask and config.MASK_BEFORE_TRANSLATE
    tr = NllbTranslator()
    job = JobProgress(available)
    log.info("Job: %d split(s) %s | %s source chars to translate | output -> %s",
             len(available), available, fmt_int(job.total_chars), config.OUTPUT_DIR)
    for split in available:
        run_split(split, present[split], tr, mask, job, limit=args.limit)
    log.info("ALL DONE: %s src-chars translated this run | total job time %s",
             fmt_int(job.run_chars), human_time(job.elapsed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
