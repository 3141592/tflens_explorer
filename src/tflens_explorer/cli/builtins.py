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


def handle_prompt_run(context: CommandContext) -> None:
    model = context.session.model
    prompt = context.session.current_prompt
    if model is None:
        print("No model loaded.")
        return

    from tflens_explorer.services.model_service import prompt_run

    results = prompt_run(model, prompt)

    print(results)
