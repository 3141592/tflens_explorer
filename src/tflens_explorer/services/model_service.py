"""Model loading service."""

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources.transformers import list_supported_models

def list_models(partial_name):
    models = list_supported_models()
    model_list = []

    for item in models:
        if partial_name == "":
            model_list.append(item)
        elif partial_name in item:
            model_list.append(item)

    return model_list


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


def tokens(model, prompt):
    # tensor([[50256,   464,  3290,  3332,   319,   262]], device='cuda:0')
    tokens = model.to_tokens(prompt)

    token_list = []
    token_list.append("prepend_bos=True")
    for index, token in enumerate(tokens[0]):
        #print(f"index: {index}, token: {token}")
        str_token = model.to_str_tokens(token)
        token_line = f"[{index}] {token} -> '{str_token[0]}'"
        token_list.append(token_line)

    return token_list