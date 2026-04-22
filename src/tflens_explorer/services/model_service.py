"""Model loading service."""

from transformer_lens.model_bridge import TransformerBridge


def load_model(model_name: str, device: str = "cuda"):
    return TransformerBridge.boot_transformers(model_name, device=device)


def get_model_info(model) -> dict:
    info = {
        "object_type": type(model).__name__,
    }

    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for field_name in [
            "model_name",
            "n_layers",
            "n_heads",
            "d_model",
            "d_head",
            "d_vocab",
            "n_ctx",
            "device",
        ]:
            info[field_name] = getattr(cfg, field_name, "unknown")

    return info

def prompt_run(model, prompt, new_tokens):
    return model.generate(prompt, max_new_tokens=new_tokens, do_sample=False)