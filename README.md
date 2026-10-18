# T2I-DFLogic

**T2I Doesn't Follow Logic.** This repo generates images for three logic-constraint prompt sets with
14 text-to-image models, timing every image so the models can be ranked by speed.

```
prompts/        the three vetted prompt files (input)
generate.py     generate images with one model (resumable, shardable)
summarize.py    rank models by measured time per prompt file, fastest first
t2i_dflogic/    data loading, the run loop, and one wrapper per model
envs/setup.sh   builds the conda envs and clones the model repos at pinned commits
slurm/          SOL job script and submit helper
```

## Prompt files

| Key | File | Prompts |
|---|---|---|
| `pl` | `propositional_logic_400_vetted.json` | 400 |
| `fol` | `first_order_logic_450_vetted.json` | 440 (vetting dropped 10 of the 450) |
| `nmr` | `non_monotonic_reasoning_400_vetted.json` | 400 |

Every model gets one image per prompt, 1,240 in total. Each prompt is passed to the model unchanged.

## Models

| Key | Model | Family | Env | Output size | Sampling (from the model's own example) |
|---|---|---|---|---|---|
| `think-then-generate` | Think-Then-Generate | CoT | diffusers | 1024 | reasoning up to 1024 tokens, then 50 steps, true CFG 4 |
| `bagel-zebra-cot` | Bagel-Zebra-CoT | CoT | bagel | 1024 | up to 4 reasoning steps with sketches, then the final image |
| `qwen-image` | Qwen-Image | Unified | diffusers | 1328 | 50 steps, true CFG 4 |
| `lumina-dimoo` | Lumina-DiMOO | Unified | dimoo | 1024 | 64 steps, CFG 4 |
| `show-o2-7b` | Show-o2-7B | Unified | showo2 | 432 | 50 steps, CFG 7.5 |
| `omnigen2` | OmniGen2 | Unified | omnigen2 | 1024 | 50 steps, text CFG 4 |
| `bagel-7b-mot` | BAGEL-7B-MoT | Unified | bagel | 1024 | 50 steps, CFG 4, thinking off |
| `janus-pro-7b` | Janus-Pro-7B | Unified | janus | 384 | 576 tokens, CFG 5 |
| `sd3.5-large` | SD 3.5 Large | Diffusion | diffusers | 1024 | 28 steps, CFG 3.5 (gated) |
| `flux2-klein-4b` | FLUX.2 [klein] 4B | Diffusion | diffusers | 1024 | 4 steps |
| `flux2-klein-9b` | FLUX.2 [klein] 9B | Diffusion | diffusers | 1024 | 4 steps (gated) |
| `sana1.5-4.8b` | SANA1.5-4.8B | Diffusion | diffusers | 1024 | 20 steps, CFG 4.5 |
| `z-image-turbo` | Z-Image Turbo | Diffusion | diffusers | 1024 | 8 passes, no CFG |
| `flux1-dev` | FLUX.1 [dev] | Diffusion | diffusers | 1024 | 50 steps, guidance 3.5 (gated) |
| `kolors` | Kolors | Diffusion | diffusers | 1024 | 50 steps, CFG 5 |

`python generate.py --list-models` prints the same list with checkpoints. To change a setting for one
run, use `--set`, e.g. `--set num_inference_steps=25`. Unknown keys are rejected.

## Setup on SOL (once)

```bash
cd /scratch/$USER && git clone https://github.com/SriGoli-0123/T2I-DFLogic.git && cd T2I-DFLogic
bash envs/setup.sh all          # or one env: diffusers | omnigen2 | dimoo | showo2 | janus | bagel
echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc
```

The models disagree on torch and transformers versions, so there are six conda envs (`t2i-<env>`).
`setup.sh` follows each repo's own requirements and clones the repos into `third_party/` at the
commits the wrappers were written against.

FLUX.1 [dev], SD 3.5 Large and FLUX.2 [klein] 9B are gated. Accept their licenses on Hugging Face,
then run `hf auth login` once. Weights download on first use. To fetch them ahead of time on a login
node, run:

```bash
source activate t2i-diffusers && python generate.py --model qwen-image --download-only
```

## Running

```bash
python generate.py --model dummy --limit 3          # no GPU needed: checks the whole pipeline
slurm/submit.sh flux2-klein-4b 1 --limit 5          # first real check: 5 prompts per file
slurm/submit.sh flux2-klein-4b                      # all 1,240 prompts, one A100
slurm/submit.sh qwen-image 4                        # 4 A100s in parallel, ~310 prompts each
python summarize.py                                 # timing table, fastest first
```

`submit.sh` submits an array job with one A100 per shard. It picks the walltime from a rough
per-image estimate ×1.5. The shard counts below keep each job under about a day:

| Shards | Models |
|---|---|
| 1 | flux2-klein-4b/9b, z-image-turbo, sana1.5-4.8b, kolors, show-o2-7b, janus-pro-7b, sd3.5-large, flux1-dev, omnigen2 |
| 2 | bagel-7b-mot, lumina-dimoo |
| 4 | qwen-image, think-then-generate |
| 6 | bagel-zebra-cot |

Rerunning the same command skips finished images. If a shard times out, resubmit it and it continues
where it stopped. A job stops itself after 3 failures in a row, rather than burning its walltime;
failures are logged to `errors.jsonl` and retried on the next run. Partition, QOS and memory
defaults are in `slurm/generate.sbatch`. Override them without editing, e.g.
`SBATCH_PARTITION=public slurm/submit.sh ...`.

## Running on a Mac (Apple GPU)

FLUX.2 [klein] 4B also runs on a 16 GB Apple-silicon Mac. Its text encoder and transformer are
~8 GB each, so `--low-memory` loads them in turns. For each chunk of 100 prompts it loads the encoder,
encodes all 100 prompts, swaps the encoder out, loads the transformer and draws the 100 images.
Each image's recorded time includes its own encoding, but not the model swaps.

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.12.1 torchvision diffusers==0.40.0 transformers==5.18.0 accelerate==1.14.0
.venv/bin/python generate.py --model flux2-klein-4b --download-only     # ~16 GB
caffeinate -i .venv/bin/python generate.py --model flux2-klein-4b --low-memory
```

`caffeinate -i` keeps the Mac awake for as long as the job runs. Keep it plugged in with the lid open.

## Outputs

```
outputs/<model>/seed42/
  runs/<run_id>.json                     settings, GPU, load time, warm-up time, counts, peak memory
  <pl|fol|nmr>/images/<id>.png           the image to judge
  <pl|fol|nmr>/images/<id>_steps/        Bagel-Zebra-CoT sketches
  <pl|fol|nmr>/records/shard<i>of<n>.jsonl
```

Each record line holds:
- the instance fields (`id`, `logic_family`, `logic_category`, `combination`, `domain`, `prompt`)
- `image`, `width`, `height`, `seed`
- `seconds`, the time to make that image
- `tokens`, the prompt length against the model's text limit, with a `truncated` flag
- `reasoning` and `steps`, for the CoT models
- `info`, e.g. Think-Then-Generate's revised prompt

`summarize.py` writes `outputs/timing_summary.md` and `.csv`. For a file that is not finished yet,
its full time is projected from the images done so far.

## Choices worth knowing

- **What the timing covers.** `seconds` is the wall time of one generation, including text encoding,
  reasoning, denoising and decoding. It is measured after one untimed warm-up image and excludes
  model loading and saving the PNG.
- **Long prompts get cut off.** Some models stop reading after a fixed number of tokens:
  - Kolors at 256 (the pipeline refuses more)
  - SANA at 300
  - Show-o2-7B at 290 (from its config)

  The cut-off part is the end of the prompt, where the constraints are. `tokens.truncated` marks
  those images, and `summarize.py` counts them. SD 3.5 Large's limit was raised from 256 to 512 so
  it reads whole prompts; every other model reads at least 512 tokens.
- **Prompt changes.**
  - Bagel-Zebra-CoT has no official text-to-image mode. It gets the prefix "Generate an image that
    matches the following description." and runs its official reasoning loop, and then one last image
    is generated from the whole reasoning context.
  - Think-Then-Generate rewrites the prompt by design; the rewrite is saved.
  - Qwen-Image's optional "Ultra HD, 4K" suffix is not added.
- **Show-o2.** Its demo's NSFW checker, which can black out images, is not applied.
- **Seeds.** One seed per run (default 42). Diffusers models use a CPU generator, so the starting
  noise is the same on any GPU. For more samples per prompt, rerun with `--seed 43`, which writes
  to `seed43/`.
- **Untested on GPU so far.** The wrappers follow each model's official inference script at the
  pinned commit. Only the shared pipeline (data, sharding, resume, records, summary) has been run,
  using the dummy model. Start each model with `--limit 5` before a full run.
