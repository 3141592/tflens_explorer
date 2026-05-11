"""Prompt command handlers."""

from pathlib import Path
from tflens_explorer.core.types import CommandContext
from tflens_explorer.services.model_service import get_cache_tensor


def handle_cache_run(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return
    
    from tflens_explorer.services.model_service import cache_run
    logits, cache = cache_run(model, prompt)
    context.session.prompt = prompt
    context.session.logits = logits
    context.session.cache = cache
 
    tokens = model.to_tokens(prompt)
    context.session.tokens = tokens

    print(f"Cached forward pass for {len(tokens[0])} tokens.")
    print("Available for inspection.")


def handle_cache_show(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return
    
    lines = [
        f"prompt: {prompt}",
        f"prepend_bos={context.session.prepend_bos}",
    ]

    keys = list(context.session.cache.keys())
    lines.append(f"cache_keys = {len(keys)}")
    lines.append("first_keys:")

    for key in keys[:10]:
        lines.append(f"  {key}")

    cache_info = "\n".join(lines)
    print(cache_info)    

def handle_cache_keys(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return

    cache = context.session.cache
    if not cache:
        print("No cache set. Use: cache-run <text>")
        return

    arg = ""
    if context.args:
        arg = context.args[0]

    keys = list(context.session.cache.keys())
    for key in keys:
        if arg == "":
            print(key)
        elif arg in key:
            print(key)
        

def handle_cache_layer(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return

    cache = context.session.cache
    if not cache:
        print("No cache set. Use: cache-run <text>")
        return

    args = ""
    if context.args:
        layer = int(context.args[0])

    if isinstance(layer, int):
        print("")
    else:
        print("The argument is not an integer: cache-layer <layer #>")
        return

    if layer >= model.cfg.n_layers:
        print(f"This cache only has {model.cfg.n_layers} layers: (0 - {model.cfg.n_layers - 1}).")
        return

    keys = list(context.session.cache.keys())
    for key in keys:
        if f"blocks.{layer}" in key:
            print(key)
    print()
        

def handle_cache_tensor(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return

    cache = context.session.cache
    if not cache:
        print("No cache set. Use: cache-run <text>")
        return

    args = ""
    if context.args:
        layer = context.args[0]
    else:
        print("No layer name set. Use: cache-layer <layer #> to find layer names.")
        return

    keys = list(context.session.cache.keys())

    if is_layer_in_cache(keys, layer):
        tensor_info = get_cache_tensor(model, prompt, layer)
        for key, value in tensor_info.items():
            print(f"{key}: {value}")

    else:
        print(f"The layer name {layer} does not exist in this cache. Run: cache-layer <layer #> to find a layer name.")
        return

    print()


def is_layer_in_cache(cache_keys, layer) -> bool:
    return layer in cache_keys