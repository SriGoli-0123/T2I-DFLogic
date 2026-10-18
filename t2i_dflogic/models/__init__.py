"""Registry of model wrappers, in the order of the model table (CoT, Unified, Diffusion)."""

from .bagel import Bagel, BagelZebraCoT
from .diffusers_models import Flux1Dev, Flux2Klein4B, Flux2Klein9B, Kolors, QwenImage, Sana15, SD35Large, ZImageTurbo
from .dummy import Dummy
from .janus_pro import JanusPro
from .lumina_dimoo import LuminaDiMOO
from .omnigen2 import OmniGen2
from .show_o2 import ShowO2
from .think_then_generate import ThinkThenGenerate

MODELS = {cls.key: cls for cls in [
    ThinkThenGenerate, BagelZebraCoT,
    QwenImage, LuminaDiMOO, ShowO2, OmniGen2, Bagel, JanusPro,
    SD35Large, Flux2Klein4B, Flux2Klein9B, Sana15, ZImageTurbo, Flux1Dev, Kolors,
    Dummy,
]}


def get(key):
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose from: {', '.join(MODELS)}")
    return MODELS[key]
