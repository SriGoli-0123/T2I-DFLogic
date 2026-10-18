"""Generate one image per prompt with one model, timing each image.

Output layout (one folder per model and seed):

    outputs/<model>/seed<seed>/
        runs/<run_id>.json                    settings, GPU, load time, warm-up, counts for one job
        <dataset>/images/<id>.png             the image that gets judged
        <dataset>/images/<id>_steps/step<k>.png   CoT sketches (Bagel-Zebra-CoT only)
        <dataset>/records/shard<i>of<n>.jsonl one line per finished image (timing, tokens, reasoning)
        <dataset>/errors.jsonl                failures; those ids are retried on the next run

An id counts as done once its record line exists and its image is on disk, so a killed or
timed-out job can simply be resubmitted. Each shard appends to its own records file, so array
jobs never write to the same file.
"""

import json
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import data
from .models import get

WARMUP_PROMPT = "A photo of a red apple on a wooden table."
MAX_CONSECUTIVE_ERRORS = 3


def run_root(out_root, model_key, seed):
    return Path(out_root) / model_key / f"seed{seed}"


def read_records(ds_dir):
    records = []
    for path in sorted((ds_dir / "records").glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:  # half-written last line from a killed job
                    pass
    return records


def done_ids(ds_dir):
    return {r["id"] for r in read_records(ds_dir) if (ds_dir / r["image"]).exists()}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def resolve_device(name):
    """'auto' picks an NVIDIA GPU, else Apple's GPU (MPS), else the CPU."""
    torch = _torch()
    if name != "auto":
        return name
    if torch and torch.cuda.is_available():
        return "cuda"
    if torch and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(device):
    """GPU work runs asynchronously; wait for it so the clock measures the real generation time."""
    torch = _torch()
    if torch and device.startswith("cuda"):
        torch.cuda.synchronize()
    elif torch and device == "mps":
        torch.mps.synchronize()


def _device_name(device):
    torch = _torch()
    if torch and device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    if device == "mps":
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return f"{chip.stdout.strip() or 'Apple GPU'} (MPS)"
    return platform.processor() or "cpu"


def _save_png(image, path):
    tmp = path.with_name(path.stem + ".tmp.png")  # write then rename, so a crash never leaves half a PNG
    image.save(tmp)
    os.replace(tmp, path)


def plan(args):
    root = run_root(args.out, args.model, args.seed)
    todo = {}
    for ds in args.datasets:
        items = data.select(data.load(ds), args.shard, args.num_shards, args.ids, args.limit)
        done = done_ids(root / ds) if (root / ds).exists() else set()
        todo[ds] = [it for it in items if it.id not in done]
        print(f"{ds}: {len(items)} prompts in shard {args.shard}/{args.num_shards}, "
              f"{len(items) - len(todo[ds])} already done, {len(todo[ds])} to generate")
    return root, todo


def run(args):
    cls = get(args.model)
    device = resolve_device(args.device)
    # Built before planning so a bad --set or --low-memory fails at once, before any loading.
    gen = cls(device=device, offload=args.offload, overrides=args.overrides, low_memory=args.low_memory)
    root, todo = plan(args)
    total = sum(len(v) for v in todo.values())
    if args.dry_run or total == 0:
        return

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_shard{args.shard}of{args.num_shards}"
    torch = _torch()
    info = dict(run_id=run_id, model=cls.key, model_name=cls.name, family=cls.family, hf_id=cls.hf_id,
                env=cls.env, settings=gen.settings, seed=args.seed, datasets=list(todo),
                shard=args.shard, num_shards=args.num_shards, offload=args.offload, low_memory=args.low_memory,
                device=device, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"), gpu=_device_name(device),
                torch_version=torch.__version__ if torch else None, python=platform.python_version(),
                command=" ".join(sys.argv), started=_now())
    (root / "runs").mkdir(parents=True, exist_ok=True)
    info_path = root / "runs" / f"{run_id}.json"

    def write_info():
        info_path.write_text(json.dumps(info, indent=2) + "\n")

    write_info()
    print(f"loading {cls.name} ({cls.hf_id}) ...", flush=True)
    t0 = time.perf_counter()
    gen.load()
    info["load_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"loaded in {info['load_seconds']} s", flush=True)
    if args.warmup:
        # The first call pays one-off costs (GPU kernels, caches); keep it out of the per-image times.
        _sync(device)
        t0 = time.perf_counter()
        gen.generate(WARMUP_PROMPT, args.seed)
        _sync(device)
        info["warmup_seconds"] = round(time.perf_counter() - t0, 2)
        print(f"warm-up image took {info['warmup_seconds']} s", flush=True)
    write_info()

    info["images_done"], info["errors"], info["prepare_seconds"] = 0, 0, 0.0
    consecutive_errors = 0
    for ds, items in todo.items():
        ds_dir = root / ds
        (ds_dir / "images").mkdir(parents=True, exist_ok=True)
        (ds_dir / "records").mkdir(parents=True, exist_ok=True)
        with (ds_dir / "records" / f"shard{args.shard}of{args.num_shards}.jsonl").open("a") as records:
            prepared = {}
            for n, it in enumerate(items, 1):
                if gen.chunk_size and (n - 1) % gen.chunk_size == 0:
                    # Low-memory mode: encode the next chunk of prompts while the image model is out of memory.
                    chunk = items[n - 1:n - 1 + gen.chunk_size]
                    print(f"[{ds}] preparing prompts {n}-{n - 1 + len(chunk)} ...", flush=True)
                    t0 = time.perf_counter()
                    prepared = gen.prepare([c.prompt for c in chunk])
                    info["prepare_seconds"] = round(info["prepare_seconds"] + time.perf_counter() - t0, 2)
                try:
                    tokens = gen.token_report(it.prompt)
                    _sync(device)
                    t0 = time.perf_counter()
                    out = gen.generate(it.prompt, args.seed)
                    _sync(device)
                    # In low-memory mode the prompt's own encoding time (not the model swaps) is added back,
                    # so the time covers the same work as a normal run.
                    encode_seconds = prepared.get(it.prompt)
                    seconds = time.perf_counter() - t0 + (encode_seconds or 0)
                    image_rel = f"images/{it.id}.png"
                    _save_png(out.image, ds_dir / image_rel)
                    steps = []
                    for k, st in enumerate(out.steps, 1):
                        step_rel = None
                        if st.get("image") is not None:
                            (ds_dir / "images" / f"{it.id}_steps").mkdir(exist_ok=True)
                            step_rel = f"images/{it.id}_steps/step{k}.png"
                            _save_png(st["image"], ds_dir / step_rel)
                        steps.append({"text": st.get("text"), "image": step_rel})
                    record = dict(id=it.id, dataset=ds, domain=it.domain, logic_family=it.logic_family,
                                  logic_category=it.logic_category, combination=it.combination,
                                  prompt=it.prompt, image=image_rel, width=out.image.width,
                                  height=out.image.height, seed=args.seed, seconds=round(seconds, 3),
                                  tokens=tokens, reasoning=out.text, steps=steps or None, info=out.info or None,
                                  text_encode_seconds=None if encode_seconds is None else round(encode_seconds, 3),
                                  model=cls.key, gpu=info["gpu"], run_id=run_id, finished=_now())
                    records.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.flush()
                    info["images_done"] += 1
                    consecutive_errors = 0
                    cut = " (prompt truncated)" if tokens and tokens["truncated"] else ""
                    print(f"[{ds} {n}/{len(items)}] {it.id} {seconds:.1f} s{cut}", flush=True)
                except Exception as exc:  # noqa: BLE001 - one bad prompt must not end a long job
                    info["errors"] += 1
                    consecutive_errors += 1
                    with (ds_dir / "errors.jsonl").open("a") as f:
                        f.write(json.dumps(dict(id=it.id, run_id=run_id, error=repr(exc),
                                                traceback=traceback.format_exc(), time=_now())) + "\n")
                    print(f"[{ds} {n}/{len(items)}] {it.id} FAILED: {exc!r}", flush=True)
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        info["stopped"] = f"{MAX_CONSECUTIVE_ERRORS} failures in a row"
                        break
        write_info()
        if "stopped" in info:
            break

    info["finished"] = _now()
    if torch and device.startswith("cuda"):
        info["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    write_info()
    print(f"done: {info['images_done']} images, {info['errors']} errors; run info in {info_path}")
    if "stopped" in info:
        sys.exit(f"stopped after {info['stopped']}; see the errors.jsonl files")
