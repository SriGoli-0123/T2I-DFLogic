"""OmniGen2, through the pipeline in its own repo (third_party/OmniGen2).

Settings follow the repo's example_t2i.sh and inference.py defaults.
"""

from .base import Generator, Output, add_repo_to_path

NEGATIVE_PROMPT = ("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, "
                   "mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs "
                   "censor, censored, censor_bar")


class OmniGen2(Generator):
    key, name, family = "omnigen2", "OmniGen2", "Unified"
    env = "omnigen2"
    hf_id = "OmniGen2/OmniGen2"
    est_seconds = 25
    DEFAULTS = dict(height=1024, width=1024, num_inference_steps=50, text_guidance_scale=4.0,
                    image_guidance_scale=2.0, cfg_range=[0.0, 1.0], max_sequence_length=1024,
                    negative_prompt=NEGATIVE_PROMPT)

    def load(self):
        import torch
        add_repo_to_path("OmniGen2")
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        self.torch = torch
        pipe = OmniGen2Pipeline.from_pretrained(self.hf_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
        pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(self.hf_id, subfolder="transformer",
                                                                      torch_dtype=torch.bfloat16)
        if self.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        self.pipe = pipe

    def generate(self, prompt, seed):
        s = self.settings
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(prompt=prompt, input_images=None, width=s["width"], height=s["height"],
                           num_inference_steps=s["num_inference_steps"],
                           max_sequence_length=s["max_sequence_length"],
                           text_guidance_scale=s["text_guidance_scale"],
                           image_guidance_scale=s["image_guidance_scale"], cfg_range=tuple(s["cfg_range"]),
                           negative_prompt=s["negative_prompt"], num_images_per_prompt=1,
                           generator=generator, output_type="pil")
        return Output(result.images[0])
