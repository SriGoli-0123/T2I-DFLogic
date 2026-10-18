#!/usr/bin/env python3
"""Generate images for the vetted logic prompts with one model.

Examples:
  python generate.py --list-models
  python generate.py --model dummy --limit 3                    # laptop smoke test, no GPU
  python generate.py --model flux2-klein-4b --download-only     # fetch weights (e.g. on a login node)
  python generate.py --model flux2-klein-4b --limit 5           # first 5 prompts of each file
  python generate.py --model qwen-image --shard 2 --num-shards 8
  python generate.py --model kolors --set num_inference_steps=25
  python generate.py --model flux2-klein-4b --low-memory        # 16 GB Mac (Apple GPU)
"""

import argparse
import json
import math
import sys

from t2i_dflogic import data
from t2i_dflogic.models import MODELS, get

SLURM_SAFETY = 1.5  # walltime = estimate x this, since the per-image estimates are rough


def parse_set(pairs):
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            sys.exit(f"--set expects key=value, got {pair!r}")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


def list_models():
    print(f"{'model':22} {'family':10} {'env':9} {'~s/img':>6}  checkpoint")
    for key, cls in MODELS.items():
        gated = "  (gated: needs HF token)" if cls.gated else ""
        print(f"{key:22} {cls.family:10} {cls.env:9} {cls.est_seconds:>6}  {cls.hf_id or '-'}{gated}")


def walltime_hours(cls, datasets, num_shards):
    n = sum(len(data.load(ds)) for ds in datasets)
    return max(1, math.ceil(cls.est_seconds * SLURM_SAFETY * math.ceil(n / num_shards) / 3600 + 0.5))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="model key (see --list-models)")
    p.add_argument("--datasets", nargs="+", choices=list(data.DATASETS), default=list(data.DATASETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--ids", nargs="+", help="only these instance ids")
    p.add_argument("--limit", type=int, help="at most this many prompts per dataset (after sharding)")
    p.add_argument("--out", default="outputs")
    p.add_argument("--device", default="auto", help="cuda, mps (Apple GPU) or cpu; auto picks the first available")
    p.add_argument("--offload", action="store_true", help="CPU offload (diffusers models and OmniGen2) for small GPUs")
    p.add_argument("--low-memory", action="store_true",
                   help="load the model's parts in turns, for 16 GB machines (flux2-klein only)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="change a sampling setting, e.g. --set num_inference_steps=25")
    p.add_argument("--no-warmup", dest="warmup", action="store_false", help="skip the untimed warm-up image")
    p.add_argument("--dry-run", action="store_true", help="show what would be generated, load nothing")
    p.add_argument("--download-only", action="store_true", help="fetch the weights and exit")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--print-env", action="store_true", help="print the conda env name for --model (for scripts)")
    p.add_argument("--print-hours", action="store_true", help="print SLURM walltime hours per shard (for scripts)")
    args = p.parse_args()

    if args.list_models:
        return list_models()
    if not args.model:
        p.error("--model is required (see --list-models)")
    cls = get(args.model)
    if args.print_env:
        return print(cls.env)
    if args.print_hours:
        return print(walltime_hours(cls, args.datasets, args.num_shards))
    if args.download_only:
        return cls().download()
    args.overrides = parse_set(args.overrides)

    from t2i_dflogic.runner import run
    run(args)


if __name__ == "__main__":
    main()
