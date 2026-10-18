#!/usr/bin/env python3
"""Rank models by measured generation time per prompt file, fastest first.

Reads outputs/<model>/seed<seed>/<dataset>/records/*.jsonl. For a file that is not finished yet,
its full time is projected as (mean seconds per finished image) x (prompts in the file).
Load time and the warm-up image are not included. Writes timing_summary.md and .csv to --out.

  python summarize.py [--out outputs] [--seed 42]
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from t2i_dflogic import data
from t2i_dflogic.models import MODELS
from t2i_dflogic.runner import read_records


def fmt(seconds):
    if seconds is None:
        return "-"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def summarize(out, seed):
    sizes = {ds: len(data.load(ds)) for ds in data.DATASETS}
    rows = []
    for model_dir in sorted(p for p in Path(out).iterdir() if p.is_dir()):
        root = model_dir / f"seed{seed}"
        if not root.is_dir():
            continue
        row = dict(model=model_dir.name, done=0, cut=0, measured=0, gpus=set(), per_image=[], files={})
        for ds, n in sizes.items():
            recs = read_records(root / ds) if (root / ds).is_dir() else []
            recs = list({r["id"]: r for r in recs}.values())  # a re-generated id counts once (latest)
            secs = [r["seconds"] for r in recs]
            row["done"] += len(recs)
            row["cut"] += sum(bool(r.get("tokens") and r["tokens"]["truncated"]) for r in recs)
            row["measured"] += sum(r.get("tokens") is not None for r in recs)
            row["gpus"] |= {r["gpu"] for r in recs if r.get("gpu")}
            row["per_image"] += secs
            row["files"][ds] = (sum(secs) if len(recs) >= n else statistics.mean(secs) * n) if secs else None
        loads = [json.loads(p.read_text()).get("load_seconds") for p in (root / "runs").glob("*.json")]
        loads = [x for x in loads if x is not None]
        row["load"] = statistics.median(loads) if loads else None
        if row["per_image"]:
            row["median"] = statistics.median(row["per_image"])
            row["total"] = sum(row["files"].values()) if all(v is not None for v in row["files"].values()) else None
            rows.append(row)
    rows.sort(key=lambda r: (r["total"] is None, r["total"] or 0, r["median"]))
    return sizes, rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    sizes, rows = summarize(args.out, args.seed)
    if not rows:
        return print(f"no records under {args.out}/*/seed{args.seed}/")

    labels = {"pl": "PL", "fol": "FOL", "nmr": "NMR"}
    file_cols = [f"{labels[ds]} {n}" for ds, n in sizes.items()]
    header = ["#", "Model", "Family", "GPU", "s/image (median)", *file_cols, f"All {sum(sizes.values())}",
              "Images done", "Prompts cut off", "Load"]
    table, csv_rows = [], []
    for i, r in enumerate(rows, 1):
        cls = MODELS.get(r["model"])
        name, family = (cls.name, cls.family) if cls else (r["model"], "?")
        cut = f"{r['cut']}/{r['measured']}" if r["measured"] else "n/a"
        files = [r["files"][ds] for ds in sizes]
        table.append([str(i), name, family, ", ".join(sorted(r["gpus"])) or "-", f"{r['median']:.1f}",
                      *map(fmt, files), fmt(r["total"]), f"{r['done']}/{sum(sizes.values())}", cut, fmt(r["load"])])
        csv_rows.append([i, r["model"], name, family, ";".join(sorted(r["gpus"])), round(r["median"], 3),
                         *[None if v is None else round(v) for v in files],
                         None if r["total"] is None else round(r["total"]), r["done"], r["cut"], r["measured"],
                         r["load"]])

    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row) + " |" for row in table]
    md.append("")
    md.append("Unfinished files are projected from the mean time per finished image. "
              "'Prompts cut off' counts prompts longer than the model's text limit, out of those checked.")
    text = "\n".join(md)
    print(text)
    Path(args.out, "timing_summary.md").write_text(text + "\n")
    with open(Path(args.out, "timing_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "model", "name", "family", "gpu", "median_seconds_per_image",
                    *[f"{ds}_seconds" for ds in sizes], "all_seconds", "images_done", "prompts_cut",
                    "prompts_checked", "load_seconds"])
        w.writerows(csv_rows)


if __name__ == "__main__":
    main()
