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


def handle_model_list(context: CommandContext) -> None:
    from tflens_explorer.services.model_service import list_models
    
    arg = ""
    if context.args:
        arg = context.args[0]

    models = list_models(arg)
    for item in models:
        print(item)


def handle_model_load(context: CommandContext) -> None:
    if not context.args:
        print("Usage: model-load <model_name>")
        return

    model_name = context.args[0]

    from tflens_explorer.services.model_service import load_model

    try:
        model = load_model(model_name)
    except Exception as exc:
        print(f"Failed to load model '{model_name}': {exc}")
        return

    context.session.model = model
    context.session.current_model_name = model_name
    context.session.cache = None

    print()
    print(f"Loaded model: {model_name}")


def handle_model_info(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    from tflens_explorer.services.model_service import get_model_info

    info = get_model_info(model)

    for key, value in info.items():
        print(f"{key}: {value}")


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
