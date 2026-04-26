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
    breakpoint()
    lines = [
        f"prompt: {prompt}",
        f"prepend_bos={context.session.prepend_bos}",
    ]

    keys = list(cache.keys())
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
        



