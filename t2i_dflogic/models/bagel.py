"""BAGEL-7B-MoT and Bagel-Zebra-CoT, each through its own repo's InterleaveInferencer
(third_party/Bagel and third_party/Bagel-Zebra-CoT). Both share the t2i-bagel env.

BAGEL follows app.py's text_to_image() with thinking off (1024x1024, 50 steps, CFG text 4.0,
cfg_interval [0.4, 1.0], timestep shift 3.0, global renorm).

Bagel-Zebra-CoT was trained for interleaved text-and-sketch reasoning on visual questions and has
no official text-to-image recipe. This wrapper keeps its official loop from infz_bf16.py (system
prompt, sampling settings, one sketch per reasoning step until "Final Answer:") and adds a last
image-generation call on the whole reasoning context; that image is the one scored. The prompt is
prefixed with TASK_TEMPLATE so the model knows an image is the expected answer.
"""

import tempfile

from .base import Generator, Output, add_repo_to_path, seed_everything


def load_bagel(model_dir, checkpoint_file, vit_sizes, offload_buffers):
    """Model set-up shared by Bagel/app.py and Bagel-Zebra-CoT/infz_bf16.py (single- or multi-GPU)."""
    import os

    import torch
    from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig,
                                SiglipVisionModel)
    from modeling.qwen2 import Qwen2Tokenizer

    llm_config = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_model, vae_config = load_ae(local_path=os.path.join(model_dir, "ae.safetensors"))
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
                         vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
                         latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    tokenizer, new_token_ids, _ = add_special_tokens(Qwen2Tokenizer.from_pretrained(model_dir))

    device_map = infer_auto_device_map(model, max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
                                       no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
                                       dtype=torch.bfloat16)
    same_device = ["language_model.model.embed_tokens", "time_embedder", "latent_pos_embed", "vae2llm",
                   "llm2vae", "connector", "vit_pos_embed"]
    if torch.cuda.device_count() == 1:
        first = device_map.get(same_device[0], "cuda:0")
        for k in same_device:
            device_map[k] = first if k in device_map else "cuda:0"
    else:
        first = device_map.get(same_device[0])
        for k in same_device:
            if k in device_map and first is not None:
                device_map[k] = first
    model = load_checkpoint_and_dispatch(model, checkpoint=os.path.join(model_dir, checkpoint_file),
                                         device_map=device_map, offload_buffers=offload_buffers,
                                         offload_folder=tempfile.mkdtemp(prefix="bagel_offload_"),
                                         dtype=torch.bfloat16, force_hooks=True).eval()
    return InterleaveInferencer(model=model, vae_model=vae_model, tokenizer=tokenizer,
                                vae_transform=ImageTransform(1024, 512, 16), vit_transform=ImageTransform(*vit_sizes),
                                new_token_ids=new_token_ids)


def seed_bagel(seed):
    import torch
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Bagel(Generator):
    key, name, family = "bagel-7b-mot", "BAGEL-7B-MoT", "Unified"
    env = "bagel"
    hf_id = "ByteDance-Seed/BAGEL-7B-MoT"
    repo = "Bagel"
    checkpoint_file = "ema.safetensors"
    est_seconds = 60
    DEFAULTS = dict(cfg_text_scale=4.0, cfg_interval=[0.4, 1.0], timestep_shift=3.0, num_timesteps=50,
                    cfg_renorm_min=0.0, cfg_renorm_type="global", image_shapes=[1024, 1024])

    def load(self):
        from huggingface_hub import snapshot_download
        add_repo_to_path(self.repo)
        model_dir = snapshot_download(self.hf_id)
        self.inferencer = load_bagel(model_dir, self.checkpoint_file, (980, 224, 14), offload_buffers=True)

    def hyper(self):
        h = dict(self.settings)
        h["image_shapes"] = tuple(h["image_shapes"])
        return h

    def generate(self, prompt, seed):
        seed_bagel(seed)
        return Output(self.inferencer(text=prompt, think=False, **self.hyper())["image"])


ZEBRA_SYSTEM_PROMPT = ("You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of "
                       "thought. Think step by step and use visual aids to enhance your problem-solving. Provide your "
                       "final conclusion clearly in the format of \"Final Answer: <answer here>\"")
TASK_TEMPLATE = "Generate an image that matches the following description.\n\n{prompt}"


def _step_text(raw):
    """The official script keeps the text between <|im_start|> and <|im_end|>."""
    text = raw.split("<|im_end|>")[0]
    return text.split("<|im_start|>")[1] if "<|im_start|>" in text else text


class BagelZebraCoT(Bagel):
    key, name, family = "bagel-zebra-cot", "Bagel-Zebra-CoT", "CoT"
    hf_id = "multimodal-reasoning-lab/Bagel-Zebra-CoT"
    repo = "Bagel-Zebra-CoT"
    checkpoint_file = "model_bf16.safetensors"
    est_seconds = 240
    DEFAULTS = dict(do_sample=True, text_temperature=0.3, cfg_text_scale=4.0, cfg_img_scale=2.0,
                    cfg_interval=[0.0, 1.0], timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=0.0,
                    cfg_renorm_type="text_channel", max_think_token_n=1000, image_shapes=[1024, 1024],
                    max_reasoning_steps=4)

    def downloads(self):
        return [(self.hf_id, ["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"])]

    def load(self):
        from huggingface_hub import snapshot_download
        add_repo_to_path(self.repo)
        model_dir = snapshot_download(self.hf_id, allow_patterns=self.downloads()[0][1])
        self.inferencer = load_bagel(model_dir, self.checkpoint_file, (980, 512, 14), offload_buffers=False)

    def generate(self, prompt, seed):
        seed_bagel(seed)
        hyper = self.hyper()
        max_steps = hyper.pop("max_reasoning_steps")
        run = lambda context, **kw: self.inferencer.interleave_inference(
            context, system_prompt=ZEBRA_SYSTEM_PROMPT, **kw, **hyper)[0]

        context, steps = [TASK_TEMPLATE.format(prompt=prompt)], []
        for _ in range(max_steps):
            raw = run(context, understanding_output=True)
            text = _step_text(raw)
            context.append(text)
            if "Final Answer:" in raw or "<answer>" in raw:
                steps.append({"text": text, "image": None})
                break
            sketch = run(context)
            context.append(sketch)
            steps.append({"text": text, "image": sketch})
        image = run(context)
        reasoning = "\n\n".join(st["text"] for st in steps)
        info = {"reasoning_steps": len(steps), "sketches": sum(st["image"] is not None for st in steps),
                "reached_final_answer": bool(steps) and steps[-1]["image"] is None}
        return Output(image, text=reasoning, steps=steps, info=info)
