"""Show-o2-7B, using its own repo (third_party/Show-o/show-o2).

Mirrors show-o2/inference_t2i.py with the README's 7B demo command (432x432, guidance 7.5,
50 steps). Two differences: images are generated one prompt at a time, and the demo's Stable
Diffusion NSFW checker (which can black out images) is not applied.

The 7B config allows 1024 - 730 - 4 = 290 prompt tokens and silently drops the rest;
token_report records which prompts are cut.
"""

from .base import Generator, Output, add_repo_to_path, seed_everything, token_report


class ShowO2(Generator):
    key, name, family = "show-o2-7b", "Show-o2-7B", "Unified"
    env = "showo2"
    hf_id = "showlab/show-o2-7B"
    est_seconds = 12
    DEFAULTS = dict(config="configs/showo2_7b_demo_432x432.yaml", guidance_scale=7.5, num_inference_steps=50)

    def downloads(self):
        return [(self.hf_id, None), ("Qwen/Qwen2.5-7B-Instruct", ["*.json", "*.txt"]),
                ("Wan-AI/Wan2.1-T2V-14B", ["Wan2.1_VAE.pth"])]

    def load(self):
        import torch
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf
        root = add_repo_to_path("Show-o", "show-o2")
        from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
        from models.misc import get_text_tokenizer, prepare_gen_input
        from transport import Sampler, create_transport
        from utils import denorm, get_hyper_params, path_to_llm_name

        self.torch, self.denorm = torch, denorm
        self.omni_attn_mask_naive, self.prepare_gen_input = omni_attn_mask_naive, prepare_gen_input
        config = OmegaConf.load(root / self.settings["config"])
        dtype = torch.bfloat16

        vae_path = hf_hub_download("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth")
        self.vae = WanVAE(vae_pth=vae_path, dtype=dtype, device=self.device)

        llm = config.model.showo.llm_model_path
        self.tokenizer, token_ids = get_text_tokenizer(llm, add_showo_tokens=True, return_showo_token_ids=True,
                                                       llm_name=path_to_llm_name[llm])
        config.model.showo.llm_vocab_size = len(self.tokenizer)
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False)
        self.model = model.to(self.device).to(dtype).eval()

        if config.model.showo.add_time_embeds:  # the time embedding is prepended to the vision tokens
            config.dataset.preprocessing.num_t2i_image_tokens += 1
            config.dataset.preprocessing.num_mmu_image_tokens += 1
            config.dataset.preprocessing.num_video_tokens += 1
        (self.num_image_tokens, _, _, self.max_seq_len, self.max_text_len, self.latent_dim, self.patch_size,
         self.latent_w, self.latent_h, self.pad_id, self.bos_id, self.eos_id, self.boi_id, self.eoi_id,
         _, _, self.img_pad_id, _, _) = get_hyper_params(config, self.tokenizer, token_ids)

        tc = config.transport
        transport = create_transport(path_type=tc.path_type, prediction=tc.prediction, loss_weight=tc.loss_weight,
                                     train_eps=tc.train_eps, sample_eps=tc.sample_eps, snr_type=tc.snr_type,
                                     do_shift=tc.do_shift, seq_len=self.num_image_tokens)
        self.sampler, self.transport_cfg = Sampler(transport), tc

    def generate(self, prompt, seed):
        torch, s, tc = self.torch, self.settings, self.transport_cfg
        seed_everything(seed)
        guidance = s["guidance_scale"]
        text, text_null, pos, pos_null = self.prepare_gen_input(
            [prompt], self.tokenizer, self.num_image_tokens, self.bos_id, self.eos_id, self.boi_id, self.eoi_id,
            self.pad_id, self.img_pad_id, self.max_text_len, self.device)
        z = torch.randn((1, self.latent_dim, self.latent_h * self.patch_size, self.latent_w * self.patch_size))
        z = z.to(torch.bfloat16).to(self.device)
        if guidance > 0:
            z, text, pos = torch.cat([z, z]), torch.cat([text, text_null]), torch.cat([pos, pos_null])
        mask = self.omni_attn_mask_naive(text.size(0), self.max_seq_len, pos, self.device).to(torch.bfloat16)
        sample_fn = self.sampler.sample_ode(sampling_method=tc.sampling_method, num_steps=s["num_inference_steps"],
                                            atol=tc.atol, rtol=tc.rtol, reverse=tc.reverse,
                                            time_shifting_factor=tc.time_shifting_factor)
        with torch.no_grad():
            samples = sample_fn(z, self.model.t2i_generate, text_tokens=text, attention_mask=mask,
                                modality_positions=pos, output_hidden_states=True, max_seq_len=self.max_seq_len,
                                guidance_scale=guidance)[-1]
            if guidance > 0:
                samples = torch.chunk(samples, 2)[0]
            images = self.vae.batch_decode(samples.unsqueeze(2)).squeeze(2)
        from PIL import Image
        return Output(Image.fromarray(self.denorm(images)[0]))

    def token_report(self, prompt):
        n = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return token_report(n, self.max_text_len)
