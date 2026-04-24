"""Built-in command handlers."""

from tflens_explorer.core.types import CommandContext


def handle_help(context: CommandContext) -> None:
    print("Available commands:")
    for command in context.registry.unique_commands():
        aliases = f" (aliases: {', '.join(command.aliases)})" if command.aliases else ""
        print(f"  {command.name}{aliases} - {command.description}")


def handle_commands(context: CommandContext) -> None:
    for command in context.registry.unique_commands():
        print(command.name)


def handle_quit(context: CommandContext) -> None:
    context.session.running = False


def handle_prompt_set(context: CommandContext) -> None:
    if not context.args:
        print("Usage: prompt-set <text>")
        return

    context.session.current_prompt = " ".join(context.args)
    print("Prompt updated.")


def handle_prompt_show(context: CommandContext) -> None:
    if not context.session.current_prompt:
        print("(no prompt set)")
        return

    print(context.session.current_prompt)


def parse_new_tokens(args):
    for arg in args:
        if arg.startswith("new_tokens="):
            return int(arg.split("=")[1])
    return 10


def handle_prompt_run(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return
    
    from tflens_explorer.services.model_service import prompt_run

    n = parse_new_tokens(context.args)

    output = prompt_run(model, prompt, new_tokens=n)
    context.session.last_output = output
    print(output)


def handle_tokens(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return
    
    from tflens_explorer.services.model_service import tokens
    token_list = tokens(model, prompt)

    for token in token_list:
        print(token)


def handle_logits(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return
    
    from tflens_explorer.services.model_service import logits
    logits_list = logits(model, prompt)

    for logit in logits_list:
        print(logit)


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
    breakpoint()
    print(cache_info)

