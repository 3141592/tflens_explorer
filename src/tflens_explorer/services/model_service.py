"""Model loading service."""

import os
import torch
from pathlib import Path
from time import perf_counter
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources.transformers import list_supported_models
from tflens_explorer.config.config_loader import load_model_aliases

MODEL_ALIASES = load_model_aliases()


def has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN"))

def try_hf_login(verbose: bool = True) -> bool:
    token = os.environ.get("HF_TOKEN")

    if not token:
        if verbose:
            print("HF login: skipped, no HF_TOKEN")
        return False

    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        if verbose:
            print("HF login: succeeded")
        return True
    except Exception as e:
        if verbose:
            print(f"HF login: failed ({type(e).__name__}: {e})")
        return False


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def is_model_cached(model_name) -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        return False

    model_dirs = sorted(cache_dir.glob("models--*"))

    if not model_dirs:
        return False

    for model_dir in model_dirs:
        model_id = model_dir.name.replace("models--", "").replace("--", "/")
        if model_name in model_id:
            return True
    return False

def timed(label, fn):
    start = perf_counter()
    result = fn()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.2f}s")
    return result


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
    cached = is_model_cached(model_name)

    if cached:
        print("Cache check: local model found")
        print("HF mode: offline/cache-only")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        print("Cache check: local model not found")
        print("HF mode: online/download allowed")
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        try_hf_login()

    model = timed(
        "Load model weights",
        lambda: TransformerBridge.boot_transformers(model_name, device=device),
    )

    return model


def load_quantized_model(model_name: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # or load_in_8bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    import torch
    import bitsandbytes as bnb

    print(type(model))

    for name, module in model.named_modules():
        if "Linear4bit" in type(module).__name__ or "Linear8bit" in type(module).__name__:
            print(name, type(module))
            break
    else:
        print("No bitsandbytes quantized Linear modules found")

    return model


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


def model_show(model):
    print(model.cfg.model_name)
    print()
    print(model.named_children)


def prompt_run(model, prompt, new_tokens):
    #tokenizer = model.tokenizer
    #print(type(tokenizer))
    return model.generate(prompt, max_new_tokens=new_tokens, do_sample=False)


def tokens(model, prompt, prepend_bos):
    # tensor([[50256,   464,  3290,  3332,   319,   262]], device='cuda:0')
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)

    token_list = []
    token_list.append(f"tokens[0].shape: {tokens[0].shape}")
    token_list.append(f"prepend_bos={prepend_bos}")
    for index, token in enumerate(tokens[0]):
        #print(f"index: {index}, token: {token}")
        str_token = model.to_str_tokens(token)
        token_line = f"[{index}] {token} -> '{str_token[0]}'"
        token_list.append(token_line)

    return token_list


def token_decode(model, token_id):
    if token_id < 0 or token_id >= model.cfg.d_vocab:
        return (-1)

    decoded_token = model.tokenizer.decode(token_id)
    return decoded_token

def token_encode(model, str_token):
    token_id = model.tokenizer.encode(str_token)
    return token_id

def logits(model, prompt, prepend_bos):
    logits = model(prompt, prepend_bos=prepend_bos)

    logits_list = []
    logits_list.append(f"logits.shape: {logits.shape}")
    logits_list.append(f"prepend_bos={prepend_bos}")
    final_logits = logits[0, -1]

    # values=tensor([-82.2124, -82.5013, -82.8357, -83.0191, -83.1987, -83.3259, -83.4433, -83.5148, -83.5338, -83.6343], device='cuda:0',
    #               grad_fn=<TopkBackward0>),
    # indices=tensor([ 4314,  2323,  3996, 18507,   736,  5743,  1735,  2166, 34902,  7624],
    #               device='cuda:0'))
    final_probs = torch.softmax(final_logits, dim=-1)
    topk_probs = torch.topk(final_probs, 10)
    topk_logits = torch.topk(final_logits, 10)
    
    logits_list.append("\n VALUES")
    for index in range(len(topk_logits.values)):
        str_token = model.to_str_tokens(topk_logits.indices[index])
        value = topk_logits.values[index]
        #print(f"token: {str_token}, value: {value}")
        logit_line = f"[{index}] {value:.2f} -> '{str_token[0]}'"
        logits_list.append(logit_line)

    logits_list.append("\n PROBABILITIES")

    for index in range(len(topk_probs.values)):
        str_token = model.to_str_tokens(topk_probs.indices[index])
        value = topk_probs.values[index]
        logit_line = f"[{index}] {value:.2f} -> '{str_token[0]}'"
        logits_list.append(logit_line)

    return logits_list


def cache_run(model, prompt):
    cache = model.run_with_cache(prompt)
    return cache


def cache_show():
    return "cache_show()"
    
