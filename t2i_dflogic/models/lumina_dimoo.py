"""Lumina-DiMOO (discrete diffusion over VQ tokens), using its own repo (third_party/Lumina-DiMOO).

Mirrors inference/inference_t2i.py with the README's "normal sampling" settings: 64 steps, CFG 4.0,
temperature 1.0, ML-Cache off (set use_cache=true to turn it on, about 1.8x faster).
"""

from .base import Generator, Output, add_repo_to_path


class LuminaDiMOO(Generator):
    key, name, family = "lumina-dimoo", "Lumina-DiMOO", "Unified"
    env = "dimoo"
    hf_id = "Alpha-VLLM/Lumina-DiMOO"
    est_seconds = 60
    DEFAULTS = dict(height=1024, width=1024, timesteps=64, cfg_scale=4.0, temperature=1.0,
                    use_cache=False, cache_ratio=0.9, warmup_ratio=0.3, refresh_interval=5)

    def load(self):
        import torch
        from transformers import AutoTokenizer
        add_repo_to_path("Lumina-DiMOO")
        from config import SPECIAL_TOKENS
        from diffusers import VQModel
        from generators.image_generation_generator import generate_image
        from model import LLaDAForMultiModalGeneration
        from utils.generation_utils import setup_seed
        from utils.image_utils import add_break_line, calculate_vq_params, decode_vq_to_image
        from utils.prompt_utils import create_prompt_templates, generate_text_to_image_prompt

        self.torch = torch
        self.fns = dict(generate_image=generate_image, setup_seed=setup_seed, add_break_line=add_break_line,
                        calculate_vq_params=calculate_vq_params, decode_vq_to_image=decode_vq_to_image,
                        generate_text_to_image_prompt=generate_text_to_image_prompt)
        self.tokens = SPECIAL_TOKENS
        self.templates = create_prompt_templates()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = LLaDAForMultiModalGeneration.from_pretrained(self.hf_id, torch_dtype=torch.bfloat16,
                                                                  device_map="auto")
        self.vqvae = VQModel.from_pretrained(self.hf_id, subfolder="vqvae").to(self.device)

    def generate(self, prompt, seed):
        s, f, t = self.settings, self.fns, self.tokens
        f["setup_seed"](seed)
        height, width = s["height"], s["width"]
        seq_len, newline_every, grid_h, grid_w = f["calculate_vq_params"](height, width)
        input_prompt, uncon_prompt = f["generate_text_to_image_prompt"](prompt, self.templates)
        con_ids = self.tokenizer(input_prompt)["input_ids"]
        uncon_ids = self.tokenizer(uncon_prompt)["input_ids"]
        image_mask = f["add_break_line"]([t["mask_token"]] * seq_len, grid_h, grid_w, new_number=t["newline_token"])
        image_slots = [t["answer_start"], t["boi"]] + image_mask + [t["eoi"], t["answer_end"]]
        prompt_ids = self.torch.tensor(con_ids + image_slots, device=self.device).unsqueeze(0)
        uncon = self.torch.tensor(uncon_ids, device=self.device).unsqueeze(0)
        vq_tokens = f["generate_image"](
            self.model, prompt_ids, seq_len=seq_len, newline_every=newline_every, timesteps=s["timesteps"],
            temperature=s["temperature"], cfg_scale=s["cfg_scale"], uncon_ids=uncon, code_start=len(con_ids) + 2,
            use_cache=s["use_cache"], cache_ratio=s["cache_ratio"], refresh_interval=s["refresh_interval"],
            warmup_ratio=s["warmup_ratio"])
        image = f["decode_vq_to_image"](vq_tokens, None, vae_ckpt=self.hf_id, image_height=height,
                                        image_width=width, vqvae=self.vqvae)
        return Output(image)
