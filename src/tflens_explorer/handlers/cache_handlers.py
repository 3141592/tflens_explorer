"""Prompt command handlers."""

from pathlib import Path
from tflens_explorer.core.types import CommandContext


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
    
    from tflens_explorer.services.model_service import cache_run
    cache = context.session.cache
    if not cache:
        cache_run(model, prompt)
    
    cache_info = "prompt:" + prompt + "\n"
    cache_info += "prepend_bos=True" + "\n"
    keys = list(cache.keys())
    num_keys = len(keys)
    first_keys = keys[:10]
    cache_info += f"cache_keys = {num_keys}" + "\n"
    cache_info += f"first_keys: "
    for key in first_keys:
        cache_info += f"  {key}\n"

    #breakpoint()
    print(cache_info)

