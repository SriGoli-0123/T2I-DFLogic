"""Models that run through a stock diffusers pipeline. All share the t2i-diffusers env.

Settings follow each model card's example. The one deliberate change is
max_sequence_length=512 for SD 3.5 Large (diffusers defaults to 256), so the end of a long
prompt, where the constraints sit, is not cut off.
"""

from .base import Generator, Output, token_report


class DiffusersGenerator(Generator):
    env = "diffusers"
    pipeline_cls = None      # class name in diffusers
    dtype = "bfloat16"
    load_kwargs = {}
    tokenizer_attr = None    # pipeline tokenizer whose length cutoff matters
    chat_template = None     # apply_chat_template kwargs, for pipelines that wrap the prompt in one

    def load(self):
        import diffusers
        import torch
        self.torch = torch
        cls = getattr(diffusers, self.pipeline_cls)
        self.pipe = cls.from_pretrained(self.hf_id, torch_dtype=getattr(torch, self.dtype), **self.load_kwargs)
        self.after_load()
        if self.offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def download(self):
        # Only the component folders the pipeline loads: repos such as FLUX.2 [klein] and FLUX.1 [dev]
        # also hold a duplicate single-file checkpoint that a full snapshot would fetch too.
        import diffusers
        from huggingface_hub import snapshot_download
        print(f"downloading {self.hf_id} (pipeline components only)", flush=True)
        variant = {k: v for k, v in self.load_kwargs.items() if k == "variant"}
        getattr(diffusers, self.pipeline_cls).download(self.hf_id, **variant)
        for repo, patterns in self.downloads()[1:]:
            print(f"downloading {repo}" + (f" ({', '.join(patterns)})" if patterns else ""), flush=True)
            snapshot_download(repo, allow_patterns=patterns)

    def after_load(self):
        pass

    def generate(self, prompt, seed):
        # A CPU generator gives the same starting noise on any GPU type.
        generator = self.torch.Generator("cpu").manual_seed(seed)
        return Output(self.pipe(prompt=prompt, generator=generator, **self.settings).images[0])

    def token_report(self, prompt):
        if self.tokenizer_attr is None:
            return None
        tok = getattr(self.pipe, self.tokenizer_attr)
        text = prompt
        if self.chat_template is not None:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, **self.chat_template)
        return token_report(len(tok(text).input_ids), self.settings["max_sequence_length"])


class Flux2Klein4B(DiffusersGenerator):
    key, name, family = "flux2-klein-4b", "FLUX.2 [klein] 4B", "Diffusion"
    hf_id = "black-forest-labs/FLUX.2-klein-4B"
    pipeline_cls = "Flux2KleinPipeline"
    est_seconds = 2
    tokenizer_attr, chat_template = "tokenizer", {"enable_thinking": False}
    DEFAULTS = dict(height=1024, width=1024, num_inference_steps=4, guidance_scale=1.0, max_sequence_length=512)

    # --low-memory: the text encoder (Qwen3-4B, ~8 GB) and the transformer (~8 GB) take turns in memory,
    # so the model runs on a 16 GB machine. Per chunk: load the encoder, encode every prompt, free it,
    # reload the transformer, draw the images. The pipeline gets the embeddings from the same function
    # and settings it uses internally, so the images should match the normal mode.
    supports_low_memory = True
    LOW_MEMORY_CHUNK = 100   # embeddings held at once: 100 x 512 tokens x 7680 x bf16 = ~0.8 GB

    def load(self):
        if not self.low_memory:
            return super().load()
        import diffusers
        import torch
        self.torch, self.diffusers = torch, diffusers
        self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(self.hf_id, text_encoder=None, transformer=None,
                                                                 torch_dtype=torch.bfloat16)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.embeds = {}
        self.chunk_size = self.LOW_MEMORY_CHUNK

    def _free(self):
        import gc
        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()
        elif self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def _sync(self):
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def prepare(self, prompts):
        import time

        from transformers import AutoModelForCausalLM
        torch = self.torch
        self.pipe.transformer = None
        self._free()
        encoder = AutoModelForCausalLM.from_pretrained(self.hf_id, subfolder="text_encoder", dtype=torch.bfloat16)
        encoder.to(self.device).eval()
        seconds = {}
        with torch.inference_mode():
            for prompt in prompts:
                self._sync()
                t0 = time.perf_counter()
                self.embeds[prompt] = self.diffusers.Flux2KleinPipeline._get_qwen3_prompt_embeds(
                    text_encoder=encoder, tokenizer=self.pipe.tokenizer, prompt=prompt, dtype=torch.bfloat16,
                    device=self.device, max_sequence_length=self.settings["max_sequence_length"])
                self._sync()
                seconds[prompt] = time.perf_counter() - t0
        del encoder
        self._free()
        self.pipe.transformer = self.diffusers.Flux2Transformer2DModel.from_pretrained(
            self.hf_id, subfolder="transformer", torch_dtype=torch.bfloat16).to(self.device)
        return seconds

    def generate(self, prompt, seed):
        if not self.low_memory:
            return super().generate(prompt, seed)
        if prompt not in self.embeds:  # e.g. the warm-up prompt
            self.prepare([prompt])
        settings = {k: v for k, v in self.settings.items() if k != "max_sequence_length"}
        generator = self.torch.Generator("cpu").manual_seed(seed)
        image = self.pipe(prompt_embeds=self.embeds.pop(prompt), generator=generator, **settings).images[0]
        return Output(image)


class Flux2Klein9B(Flux2Klein4B):
    key, name = "flux2-klein-9b", "FLUX.2 [klein] 9B"
    hf_id = "black-forest-labs/FLUX.2-klein-9B"
    gated = True
    est_seconds = 3


class ZImageTurbo(DiffusersGenerator):
    key, name, family = "z-image-turbo", "Z-Image Turbo", "Diffusion"
    hf_id = "Tongyi-MAI/Z-Image-Turbo"
    pipeline_cls = "ZImagePipeline"
    load_kwargs = {"low_cpu_mem_usage": False}
    est_seconds = 3
    tokenizer_attr, chat_template = "tokenizer", {"enable_thinking": True}
    # 9 scheduler steps = 8 transformer passes; Turbo is distilled to run without guidance.
    DEFAULTS = dict(height=1024, width=1024, num_inference_steps=9, guidance_scale=0.0, max_sequence_length=512)


class Sana15(DiffusersGenerator):
    key, name, family = "sana1.5-4.8b", "SANA1.5-4.8B", "Diffusion"
    hf_id = "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers"
    pipeline_cls = "SanaPipeline"
    est_seconds = 5
    tokenizer_attr = "tokenizer"
    DEFAULTS = dict(height=1024, width=1024, num_inference_steps=20, guidance_scale=4.5, max_sequence_length=300)

    def after_load(self):
        self.pipe.text_encoder.to(self.torch.bfloat16)


class Kolors(DiffusersGenerator):
    key, name, family = "kolors", "Kolors", "Diffusion"
    hf_id = "Kwai-Kolors/Kolors-diffusers"
    pipeline_cls = "KolorsPipeline"
    dtype = "float16"
    load_kwargs = {"variant": "fp16"}
    est_seconds = 8
    tokenizer_attr = "tokenizer"
    # The pipeline refuses max_sequence_length > 256, so long prompts are cut; token_report logs it.
    DEFAULTS = dict(height=1024, width=1024, num_inference_steps=50, guidance_scale=5.0, negative_prompt="",
                    max_sequence_length=256)


class Flux1Dev(DiffusersGenerator):
    key, name, family = "flux1-dev", "FLUX.1 [dev]", "Diffusion"
    hf_id = "black-forest-labs/FLUX.1-dev"
    pipeline_cls = "FluxPipeline"
    gated = True
    est_seconds = 30
    tokenizer_attr = "tokenizer_2"   # T5; the 77-token CLIP encoder only supplies a pooled vector
    DEFAULTS = dict(height=1024, width=1024, num_inference_steps=50, guidance_scale=3.5, max_sequence_length=512)


class SD35Large(DiffusersGenerator):
    key, name, family = "sd3.5-large", "SD 3.5 Large", "Diffusion"
    hf_id = "stabilityai/stable-diffusion-3.5-large"
    pipeline_cls = "StableDiffusion3Pipeline"
    gated = True
    est_seconds = 20
    tokenizer_attr = "tokenizer_3"   # T5 carries the full prompt; CLIP stops at 77 tokens
    DEFAULTS = dict(height=1024, width=1024, num_inference_steps=28, guidance_scale=3.5, max_sequence_length=512)


class QwenImage(DiffusersGenerator):
    key, name, family = "qwen-image", "Qwen-Image", "Unified"
    hf_id = "Qwen/Qwen-Image"
    pipeline_cls = "QwenImagePipeline"
    est_seconds = 150
    tokenizer_attr = "tokenizer"
    # Model card 1:1 size. Its "Ultra HD, 4K, cinematic composition." suffix is not appended,
    # so every model sees the prompt text unchanged.
    DEFAULTS = dict(height=1328, width=1328, num_inference_steps=50, true_cfg_scale=4.0, negative_prompt=" ",
                    max_sequence_length=512)
