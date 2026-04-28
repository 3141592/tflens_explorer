"""Built-in command handlers."""

import os
from dataclasses import asdict
from tflens_explorer.core.types import CommandContext


def handle_help(context: CommandContext) -> None:
    print("Available commands:")
    for command in context.registry.unique_commands():
        aliases = f" (aliases: {', '.join(command.aliases)})" if command.aliases else ""
        print(f"  {command.name}{aliases} - {command.description}")


def handle_commands(context: CommandContext) -> None:
    arg = context.args
    for command in context.registry.unique_commands():
        if len(context.args) == 0:
            print(command.name)
        elif arg[0] in command.name:
            print(command.name)


def handle_clear(context: CommandContext) -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


def handle_quit(context: CommandContext) -> None:
    context.session.running = False


def handle_show_context(context: CommandContext) -> None:
    print("Current context:")
    for k, v in context.session:
        if k == 'model':
            continue
        print(k, v)


def clear_sessions(context: CommandContext) -> None:
    context.session.cache = None
    context.session.logits = None
    context.session.tokens = None
    context.session.scratch.clear()
    context.session.last_output = ""

    context.session.current_prompt = ""
    context.session.model = None
    context.session.current_model_name = None
    context.session.prepend_bos = None

    print("Session state cleared.")
