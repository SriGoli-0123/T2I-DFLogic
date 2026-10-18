"""Janus-Pro-7B (autoregressive, 576 image tokens at 384x384), using the janus package from
third_party/Janus (installed with pip install -e).

Mirrors the README's Janus-Pro text-to-image example (CFG 5, temperature 1) with one image per
prompt instead of 16.
"""

from .base import Generator, Output, seed_everything


class JanusPro(Generator):
    key, name, family = "janus-pro-7b", "Janus-Pro-7B", "Unified"
    env = "janus"
    hf_id = "deepseek-ai/Janus-Pro-7B"
    est_seconds = 15
    DEFAULTS = dict(cfg_weight=5.0, temperature=1.0, image_token_num_per_image=576, img_size=384, patch_size=16)

    def load(self):
        import torch
        from janus.models import VLChatProcessor
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.processor = VLChatProcessor.from_pretrained(self.hf_id)
        model = AutoModelForCausalLM.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = model.to(torch.bfloat16).to(self.device).eval()

    def generate(self, prompt, seed):
        torch, s, proc, mm = self.torch, self.settings, self.processor, self.model
        seed_everything(seed)
        conversation = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
        sft = proc.apply_sft_template_for_multi_turn_prompts(conversations=conversation,
                                                             sft_format=proc.sft_format, system_prompt="")
        input_ids = torch.LongTensor(proc.tokenizer.encode(sft + proc.image_start_tag))
        n_img, size, patch = s["image_token_num_per_image"], s["img_size"], s["patch_size"]

        with torch.inference_mode():
            # Row 0 is conditional, row 1 the unconditional (prompt padded out) for CFG.
            tokens = torch.zeros((2, len(input_ids)), dtype=torch.int, device=self.device)
            tokens[:] = input_ids
            tokens[1, 1:-1] = proc.pad_id
            inputs_embeds = mm.language_model.get_input_embeddings()(tokens)
            generated = torch.zeros((1, n_img), dtype=torch.int, device=self.device)
            past = None
            for i in range(n_img):
                out = mm.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past)
                past = out.past_key_values
                logits = mm.gen_head(out.last_hidden_state[:, -1, :])
                logits = logits[1::2] + s["cfg_weight"] * (logits[0::2] - logits[1::2])
                next_token = torch.multinomial(torch.softmax(logits / s["temperature"], dim=-1), num_samples=1)
                generated[:, i] = next_token.squeeze(dim=-1)
                next_token = torch.cat([next_token, next_token], dim=1).view(-1)
                inputs_embeds = mm.prepare_gen_img_embeds(next_token).unsqueeze(dim=1)
            dec = mm.gen_vision_model.decode_code(generated, shape=[1, 8, size // patch, size // patch])

        import numpy as np
        from PIL import Image
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        return Output(Image.fromarray(np.clip((dec[0] + 1) / 2 * 255, 0, 255).astype(np.uint8)))
