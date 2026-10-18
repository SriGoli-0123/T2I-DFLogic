#!/bin/bash
# Build one conda env for T2I-DFLogic and clone the model repo(s) it needs into third_party/.
#
#   bash envs/setup.sh <env>      env: diffusers | omnigen2 | dimoo | showo2 | janus | bagel | all
#
# Each model family pins different torch/transformers versions, so they cannot share one env:
#   t2i-diffusers  FLUX.2 [klein], Z-Image Turbo, SANA1.5, Kolors, FLUX.1 [dev], SD 3.5 Large,
#                  Qwen-Image, Think-Then-Generate
#   t2i-omnigen2   OmniGen2            t2i-dimoo   Lumina-DiMOO      t2i-showo2  Show-o2-7B
#   t2i-janus      Janus-Pro-7B        t2i-bagel   BAGEL-7B-MoT and Bagel-Zebra-CoT
#
# Versions follow each repo's own requirements. Run on a SOL login node; flash-attn (showo2,
# bagel) needs the CUDA module below and builds from source if no prebuilt wheel matches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TP="${T2I_THIRD_PARTY:-$REPO_ROOT/third_party}"
CUDA_MODULE="${CUDA_MODULE:-cuda-12.1.1-gcc-12.1.0}"
FLASH_ATTN="flash-attn==2.7.4.post1"

if command -v module >/dev/null 2>&1; then
  module load mamba/latest
fi

clone() {  # clone <dir name> <url> <commit>: the commits the wrappers in t2i_dflogic/models were written against
  local dir="$TP/$1"
  mkdir -p "$TP"
  [ -d "$dir/.git" ] || git clone "$2" "$dir"
  git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$3"
  echo "$1 at $(git -C "$dir" rev-parse --short HEAD)"
}

make_env() {  # make_env <short name> <python version>
  local name="t2i-$1"
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo "conda env $name exists; installing into it"
  else
    mamba create -y -n "$name" python="$2"
  fi
  # shellcheck disable=SC1091
  source activate "$name"
  python -m pip install --upgrade pip
}

flash_attn() {
  if command -v module >/dev/null 2>&1; then module load "$CUDA_MODULE"; fi
  pip install "$FLASH_ATTN" --no-build-isolation
}

setup_diffusers() {
  make_env diffusers 3.11
  pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install diffusers==0.40.0 transformers==5.18.0 accelerate==1.14.0 sentencepiece protobuf ftfy pillow
}

setup_omnigen2() {
  clone OmniGen2 https://github.com/VectorSpaceLab/OmniGen2.git 18e6f9d5271b517fcb32e999f10df943ae9b8f20
  make_env omnigen2 3.11
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  # The repo leaves diffusers unpinned; 0.34.0 matches its transformers==4.51.3 pin.
  pip install -r "$TP/OmniGen2/requirements.txt" diffusers==0.34.0
}

setup_dimoo() {
  clone Lumina-DiMOO https://github.com/Alpha-VLLM/Lumina-DiMOO.git 81344e60545ae8d70f11039d3450717d974f831b
  make_env dimoo 3.10
  pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r "$TP/Lumina-DiMOO/requirements.txt"
}

setup_showo2() {
  clone Show-o https://github.com/showlab/Show-o.git 45a5a2de01d1ebd10cd5864d29310a76476cdf23
  make_env showo2 3.10
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  # show-o2/build_env.sh minus the packages only its training and evaluation code uses
  # (tensorflow, deepspeed, onnx, segment_anything, video I/O, debuggers).
  pip install transformers==4.47.0 diffusers==0.31.0 einops==0.8.0 decord sentencepiece ftfy regex tqdm \
    omegaconf torchdiffeq lightning==2.4.0 wandb accelerate==0.23.0 timm==1.0.12 huggingface-hub==0.24.0 \
    dill pandas pyarrow==11.0.0 jsonlines git+https://github.com/openai/CLIP.git
  flash_attn
}

setup_janus() {
  clone Janus https://github.com/deepseek-ai/Janus.git 1daa72fa409002d40931bd7b36a9280362469ead
  # Python 3.9: janus imports attrdict, which breaks on 3.10+.
  make_env janus 3.9
  pip install -e "$TP/Janus" torch==2.0.1 transformers==4.38.2
}

setup_bagel() {
  clone Bagel https://github.com/ByteDance-Seed/Bagel.git a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f
  clone Bagel-Zebra-CoT https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT.git bbf53c327889a1764d27989d117d1a6b73ffdd3f
  make_env bagel 3.10
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r "$TP/Bagel/requirements.txt"
  flash_attn
}

case "${1:-}" in
  diffusers|omnigen2|dimoo|showo2|janus|bagel) "setup_$1" ;;
  all) for e in diffusers omnigen2 dimoo showo2 janus bagel; do bash "$0" "$e"; done ;;
  *) echo "usage: bash envs/setup.sh <diffusers|omnigen2|dimoo|showo2|janus|bagel|all>" >&2; exit 1 ;;
esac
