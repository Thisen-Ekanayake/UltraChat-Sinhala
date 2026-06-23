#!/usr/bin/env bash
#
# setup_pod_rocm.sh — provision a ROCm GPU pod (e.g. AMD MI300X) to run the
# NLLB English->Sinhala translation stage.
#
# Creates a venv, installs a ROCm build of torch plus the translation deps, and
# verifies that torch actually sees the GPU. The ROCm wheel index must match the
# pod's ROCm runtime; override it if the default does not light up the GPU:
#
#   ROCM_INDEX=https://download.pytorch.org/whl/rocm6.4 bash scripts/setup_pod_rocm.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"
VENV="${VENV:-$ROOT_DIR/.venv}"
ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.3}"

echo "==> Creating venv at $VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel

echo "==> Installing ROCm torch from $ROCM_INDEX"
pip install -q torch --index-url "$ROCM_INDEX"

echo "==> Installing translation dependencies"
pip install -q "transformers>=4.40" sentencepiece sacremoses nltk \
  huggingface_hub pyarrow numpy tqdm

echo "==> Fetching nltk sentence tokenizer (punkt)"
python - <<'PY' || true
import nltk
for pkg in ("punkt_tab", "punkt"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        print(f"  {pkg}: {e}")
PY

echo "==> Verifying torch sees the GPU"
python - <<'PY'
import torch
hip = getattr(torch.version, "hip", None)
ok = torch.cuda.is_available()
print(f"torch={torch.__version__} hip={hip} cuda_available={ok}")
if ok:
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: torch does not see the GPU — try a different "
                     "ROCM_INDEX (e.g. rocm6.4 / rocm6.2) to match the pod's ROCm.")
PY

echo "==> Done. Activate with: source $VENV/bin/activate"
