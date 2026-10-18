"""Common interface every model wrapper implements.

Heavy libraries (torch, diffusers, the models' own repos) are imported inside load() and
generate(), so listing models or doing a dry run works in any Python without them.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = Path(os.environ.get("T2I_THIRD_PARTY", REPO_ROOT / "third_party"))


@dataclass
class Output:
    image: object                                # PIL.Image.Image
    text: str | None = None                      # reasoning text (CoT models)
    steps: list = field(default_factory=list)    # CoT steps: [{"text": str, "image": PIL image or None}]
    info: dict = field(default_factory=dict)     # extra per-image facts saved in the record


class Generator:
    key = None          # name used on the command line
    name = None         # name as written in the model table
    family = None       # "Diffusion", "Unified" or "CoT"
    env = None          # conda environment built by envs/setup.sh
    hf_id = None        # main Hugging Face checkpoint
    gated = False       # needs a Hugging Face token and an accepted license
    est_seconds = None  # rough upper bound, seconds per image on one A100 80GB (SLURM walltime only)
    DEFAULTS = {}       # sampling settings, saved in each run's info file
    supports_low_memory = False
    # When set, the runner calls prepare() on each chunk of this many prompts before generating them.
    chunk_size = None

    def __init__(self, device="cuda", offload=False, overrides=None, low_memory=False):
        if low_memory and not self.supports_low_memory:
            raise ValueError(f"{self.key} has no low-memory mode")
        self.device = device
        self.offload = offload
        self.low_memory = low_memory
        overrides = overrides or {}
        unknown = set(overrides) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"{self.key}: unknown settings {sorted(unknown)}; known: {sorted(self.DEFAULTS)}")
        self.settings = {**self.DEFAULTS, **overrides}

    def downloads(self):
        """(repo_id, allow_patterns) pairs to fetch before the first load; None means the whole repo."""
        return [(self.hf_id, None)]

    def download(self):
        """Fetch the weights ahead of time, e.g. on a login node, so GPU time is not spent downloading."""
        from huggingface_hub import snapshot_download
        for repo, patterns in self.downloads():
            print(f"downloading {repo}" + (f" ({', '.join(patterns)})" if patterns else ""), flush=True)
            snapshot_download(repo, allow_patterns=patterns)

    def load(self):
        raise NotImplementedError

    def generate(self, prompt, seed) -> Output:
        raise NotImplementedError

    def prepare(self, prompts):
        """Chunked models only: do the per-prompt work that needs a different set of weights in memory
        (e.g. text encoding), and return {prompt: seconds spent on it} so it counts toward each image."""
        raise NotImplementedError

    def token_report(self, prompt):
        """{prompt_tokens, token_limit, truncated} for models with a hard text-length cutoff, else None."""
        return None


def seed_everything(seed):
    """For repos that draw noise from the global RNGs instead of taking a generator."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def add_repo_to_path(*parts):
    """Put a cloned model repo (third_party/<parts>) first on sys.path and return its path."""
    path = THIRD_PARTY.joinpath(*parts)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is missing; run `bash envs/setup.sh <env>` to clone it")
    sys.path.insert(0, str(path))
    return path


def token_report(n_tokens, limit):
    return {"prompt_tokens": n_tokens, "token_limit": limit, "truncated": n_tokens > limit}
