"""Think-Then-Generate: the Qwen2.5-VL text encoder first reasons about and rewrites the prompt,
then the Qwen-Image DiT draws the rewritten prompt.

Follows zhijie-group/Think-Then-Generate inference.py and its Hugging Face Space (processor
from Qwen2.5-VL-7B-Instruct, greedy reasoning up to 1024 tokens, 1024x1024, 50 steps, true CFG 4).
"""

import re

from .base import Output
from .diffusers_models import DiffusersGenerator

# Copied verbatim from the official inference.py.
SYSTEM_PROMPT = """You are a Prompt Optimizer specializing in image generation models (e.g., MidJourney, Stable Diffusion). Your core task is to rewrite user-provided prompts into highly clear, easy-to-render versions.
When rewriting, prioritize the following principles:
1. Start from the user's prompt, do reasoning step by step to analyze the object or scene they want to generate.
2. Focus on describing the final visual appearance of the scene. Clarify elements like the main subject’s shape, color, and state.
3. If you are confident about what the user wants to generate, directly point it out in your explanation and the final revised prompt.
4. If technical concepts are necessary but difficult for ordinary users to understand, translate them into intuitive visual descriptions.
5. Ensure the final revised prompt is consistent with the user's intent.

After receiving the user’s prompt that needs rewriting, first explain your reasoning for optimization. Then, output the final revised prompt in the fixed format of "Revised Prompt:\n". Where the specific revised content is filled in the next line.

Prompt:""" + " \n"  # the original ends with "Prompt: " and a newline


def extract_prompt(text):
    m = re.search(r"Revised Prompt:\n(.*)", text, re.DOTALL) or re.search(r"Revised Prompt:(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


class ThinkThenGenerate(DiffusersGenerator):
    key, name, family = "think-then-generate", "Think-Then-Generate", "CoT"
    hf_id = "SJTU-DENG-Lab/Think-Then-Generate-T2I"
    processor_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    pipeline_cls = "QwenImagePipeline"
    est_seconds = 175
    tokenizer_attr = "tokenizer"
    DEFAULTS = dict(height=1024, width=1024, num_inference_steps=50, true_cfg_scale=4.0, negative_prompt=" ",
                    max_sequence_length=512, reasoning_max_new_tokens=1024)

    def downloads(self):
        return [(self.hf_id, None), (self.processor_id, ["*.json", "*.txt"])]

    def load(self):
        from transformers import AutoProcessor
        super().load()
        self.processor = AutoProcessor.from_pretrained(self.processor_id)

    def generate(self, prompt, seed):
        torch = self.torch
        settings = dict(self.settings)
        max_new_tokens = settings.pop("reasoning_max_new_tokens")
        messages = [{"role": "user", "content": SYSTEM_PROMPT + prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            ids = self.pipe.text_encoder.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        new_ids = ids[:, inputs.input_ids.shape[1]:]
        reasoning = self.processor.batch_decode(new_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0]
        revised = extract_prompt(reasoning)
        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.pipe(prompt=revised, generator=generator, **settings).images[0]
        info = {"revised_prompt": revised, "reasoning_tokens": int(new_ids.shape[1]),
                "revised_prompt_tokens": super().token_report(revised)}
        return Output(image, text=reasoning, info=info)

    def token_report(self, prompt):
        # The raw prompt goes to the reasoning LLM, which has no tight cutoff; the DiT only sees the
        # revised prompt, whose length is stored per image in info.revised_prompt_tokens.
        return None
