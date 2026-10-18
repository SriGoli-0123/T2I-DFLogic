"""A stand-in model that writes the prompt onto a blank image. It needs only Pillow, so the whole
pipeline (data, sharding, resume, records, summary) can be tested on a laptop."""

import time

from .base import Generator, Output, token_report


class Dummy(Generator):
    key, name, family = "dummy", "Dummy (test only)", "Test"
    env = "any"
    est_seconds = 1
    DEFAULTS = dict(size=256, sleep=0.0, token_limit=200)

    def downloads(self):
        return []

    def load(self):
        pass

    def generate(self, prompt, seed):
        import textwrap

        from PIL import Image, ImageDraw
        time.sleep(self.settings["sleep"])
        image = Image.new("RGB", (self.settings["size"],) * 2, "white")
        ImageDraw.Draw(image).multiline_text((4, 4), f"seed {seed}\n" + "\n".join(textwrap.wrap(prompt, 40)[:14]),
                                             fill="black")
        return Output(image)

    def token_report(self, prompt):
        return token_report(len(prompt.split()), self.settings["token_limit"])
