"""Prompt command handlers."""

from pathlib import Path
from tflens_explorer.core.types import CommandContext


def parse_kv_args(args: list[str]) -> dict:
    result = {}

    for arg in args:
        if "=" not in arg:
            continue

        key, value = arg.split("=", 1)

        # basic type coercion
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # leave as string

        result[key] = value

    return result

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

    kwargs = parse_kv_args(context.args)
    new_tokens = kwargs.get("new_tokens", 10)
    if not isinstance(new_tokens, int) or new_tokens <= 0:
        print("Error: new_tokens must be a positive integer.")
        return

    output = prompt_run(model, prompt, new_tokens=new_tokens)
    context.session.last_output = output
    print(output)


