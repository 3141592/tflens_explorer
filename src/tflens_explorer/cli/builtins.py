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
