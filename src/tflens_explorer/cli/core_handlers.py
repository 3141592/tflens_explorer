"""Built-in command handlers."""

import os
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


