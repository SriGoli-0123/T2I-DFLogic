"""Load the vetted prompt files and pick which instances a run should generate."""

import json
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Short dataset keys used on the command line and in output paths.
DATASETS = {
    "pl": "propositional_logic_400_vetted.json",
    "fol": "first_order_logic_450_vetted.json",
    "nmr": "non_monotonic_reasoning_400_vetted.json",
}


@dataclass(frozen=True)
class Instance:
    dataset: str
    id: str
    domain: str
    logic_family: str
    logic_category: str
    combination: str
    prompt: str


def load(dataset, prompts_dir=PROMPTS_DIR):
    with open(Path(prompts_dir) / DATASETS[dataset]) as f:
        doc = json.load(f)
    items = [Instance(dataset, x["id"], x["domain"], x["logic_family"], x["logic_category"],
                      x["combination"], x["prompt"]) for x in doc["instances"]]
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset}: duplicate instance ids")
    if doc.get("num_instances", len(items)) != len(items):
        raise ValueError(f"{dataset}: num_instances={doc['num_instances']} but found {len(items)}")
    return items


def select(items, shard=0, num_shards=1, ids=None, limit=None):
    """Filter by id, then take every num_shards-th item so each shard gets a mix of categories."""
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} is outside 0..{num_shards - 1}")
    if ids:
        wanted = set(ids)
        items = [it for it in items if it.id in wanted]
    items = items[shard::num_shards]
    return items[:limit] if limit else items
