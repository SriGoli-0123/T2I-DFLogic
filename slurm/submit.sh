#!/bin/bash
# Submit one model as a SLURM array job, one task (one A100) per shard.
#
#   slurm/submit.sh <model> [num_shards] [extra generate.py args...]
#
#   slurm/submit.sh flux2-klein-4b                    # one job for all 1,240 prompts
#   slurm/submit.sh qwen-image 8                      # 8 jobs of ~155 prompts each
#   slurm/submit.sh kolors 1 --datasets pl --limit 20 # quick check on 20 PL prompts
#
# Walltime per shard comes from the model's rough per-image estimate x1.5. Resubmitting the same
# command skips every image that already exists, so a timed-out shard just picks up where it stopped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAX_HOURS="${MAX_HOURS:-168}"  # SOL public QOS limit (7 days)

MODEL="${1:?usage: slurm/submit.sh <model> [num_shards] [generate.py args...]}"
SHARDS="${2:-1}"
shift $(( $# >= 2 ? 2 : 1 ))
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then module load mamba/latest >/dev/null 2>&1 || true; fi
PY="${PYTHON:-python3}"
ENV_NAME="t2i-$($PY generate.py --model "$MODEL" --print-env)"
HOURS="$($PY generate.py --model "$MODEL" --num-shards "$SHARDS" --print-hours "$@")"
if [ "$HOURS" -gt "$MAX_HOURS" ]; then
  echo "warning: $MODEL needs ~${HOURS} h per shard; capping at ${MAX_HOURS} h. Use more shards." >&2
  HOURS="$MAX_HOURS"
fi

mkdir -p "$REPO_ROOT/slurm/logs"
echo "submitting $MODEL: $SHARDS shard(s), ${HOURS} h each, env $ENV_NAME"
sbatch --job-name="t2i-$MODEL" --array="0-$((SHARDS - 1))" --time="${HOURS}:00:00" \
  --output="$REPO_ROOT/slurm/logs/%x_%A_%a.out" --error="$REPO_ROOT/slurm/logs/%x_%A_%a.err" \
  --chdir="$REPO_ROOT" "$SCRIPT_DIR/generate.sbatch" "$REPO_ROOT" "$MODEL" "$ENV_NAME" "$SHARDS" "$@"
