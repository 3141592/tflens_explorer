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

def handle_clear_runtime_state(ctx: CommandContext):
    ctx.session.clear_runtime_state(keep_model=True)
    print("Session runtime state cleared.")

def handle_session_clear(ctx: CommandContext):
    ctx.session.clear_session()
    print("Session state cleared.")