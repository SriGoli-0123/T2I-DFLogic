#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#
# One shard of one model. Submit with slurm/submit.sh, which sets the job name, array range,
# walltime, log paths and these arguments:
#   <repo root> <model> <conda env> <num shards> [extra generate.py args...]
# Override the defaults above without editing, e.g. SBATCH_PARTITION=public slurm/submit.sh ...
set -euo pipefail

REPO_ROOT="$1"; MODEL="$2"; ENV_NAME="$3"; NUM_SHARDS="$4"; shift 4
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
cd "$REPO_ROOT"
echo "job ${SLURM_JOB_ID:-local} shard $SHARD/$NUM_SHARDS on $(hostname): $MODEL ($ENV_NAME)"

module load mamba/latest
# shellcheck disable=SC1091
source activate "$ENV_NAME"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python generate.py --model "$MODEL" --shard "$SHARD" --num-shards "$NUM_SHARDS" "$@"
