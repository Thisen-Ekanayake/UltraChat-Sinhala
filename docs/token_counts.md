# UltraChat-Sinhala — Token Counts (SinLLaMA tokenizer)

_Generated 2026-06-26 10:50 · tokenizer `SinLlama_merged_bf16` (vocab 139,336)_

Tokens are **raw content tokens** — each message `content` encoded with `add_special_tokens=False` (no chat template is defined on this tokenizer) and summed per dialogue. A full SFT run with role/turn special tokens would add a small per-message overhead on top.

| split | dialogues | messages | total tokens | tokens/dialogue (mean) | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_sft | 207,831 | 1,315,386 | 248,133,641 | 1,194 | 1,096 | 1,980 | 3,050 | 11,789 |
| test_sft | 23,106 | 146,276 | 27,467,232 | 1,189 | 1,089 | 1,981 | 3,061 | 5,225 |
| train_gen | 255,974 | 1,359,686 | 227,432,690 | 888 | 797 | 1,560 | 2,363 | 15,621 |
| test_gen | 28,300 | 150,098 | 25,035,938 | 885 | 795 | 1,548 | 2,369 | 20,471 |

## Totals

| group | dialogues | total tokens |
|---|---:|---:|
| SFT (train+test) | 230,937 | 275,600,873 |
| GEN (train+test) | 284,274 | 252,468,628 |
| **Grand total** | **515,211** | **528,069,501** |
