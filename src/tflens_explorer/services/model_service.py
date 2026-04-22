"""Model loading service."""

from transformer_lens import HookedTransformer


def load_model(model_name: str):
    return HookedTransformer.from_pretrained(model_name)

