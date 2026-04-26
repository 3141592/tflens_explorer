"""Model loading service."""

import torch
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources.transformers import list_supported_models
from tflens_explorer.config.config_loader import load_model_aliases

MODEL_ALIASES = load_model_aliases()

def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


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
    #tokenizer = model.tokenizer
    #print(type(tokenizer))
    return model.generate(prompt, max_new_tokens=new_tokens, do_sample=False)


def tokens(model, prompt, prepend_bos):
    # tensor([[50256,   464,  3290,  3332,   319,   262]], device='cuda:0')
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)

    token_list = []
    token_list.append(f"prepend_bos={prepend_bos}")
    for index, token in enumerate(tokens[0]):
        #print(f"index: {index}, token: {token}")
        str_token = model.to_str_tokens(token)
        token_line = f"[{index}] {token} -> '{str_token[0]}'"
        token_list.append(token_line)

    return token_list


def logits(model, prompt, prepend_bos):
    logits = model(prompt, prepend_bos=prepend_bos)

    logits_list = []
    logits_list.append(f"prepend_bos={prepend_bos}")
    final_logits = logits[0, -1]

    # values=tensor([-82.2124, -82.5013, -82.8357, -83.0191, -83.1987, -83.3259, -83.4433, -83.5148, -83.5338, -83.6343], device='cuda:0',
    #               grad_fn=<TopkBackward0>),
    # indices=tensor([ 4314,  2323,  3996, 18507,   736,  5743,  1735,  2166, 34902,  7624],
    #               device='cuda:0'))
    topk_logits = torch.topk(final_logits, 10)

    for index in range(len(topk_logits.values)):
        str_token = model.to_str_tokens(topk_logits.indices[index])
        value = topk_logits.values[index]
        #print(f"token: {str_token}, value: {value}")
        logit_line = f"[{index}] {value:.2f} -> '{str_token[0]}'"
        logits_list.append(logit_line)

    return logits_list


def cache_run(model, prompt):
    cache = model.run_with_cache(prompt)
    return cache


def cache_show():
    return "cache_show()"
    
