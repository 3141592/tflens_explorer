"""Command registry built from YAML config."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tflens_explorer.cli import builtins
from tflens_explorer.utils.yaml_loader import load_yaml


@dataclass
class CommandDefinition:
    name: str
    aliases: list[str]
    description: str
    handler: Callable


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, CommandDefinition] = {}

    def register(self, command: CommandDefinition) -> None:
        keys = [command.name, *command.aliases]
        for key in keys:
            self._commands[key] = command

    def get(self, name: str) -> CommandDefinition | None:
        return self._commands.get(name)

    def unique_commands(self) -> list[CommandDefinition]:
        unique: dict[str, CommandDefinition] = {}
        for command in self._commands.values():
            unique[command.name] = command
        return sorted(unique.values(), key=lambda item: item.name)


def resolve_handler(handler_name: str) -> Callable:
    builtin_handlers = {
        "builtin.help": builtins.handle_help,
        "builtin.commands": builtins.handle_commands,
        "builtin.quit": builtins.handle_quit,
    }

    try:
        return builtin_handlers[handler_name]
    except KeyError as exc:
        raise ValueError(f"Unknown handler in config: {handler_name}") from exc


def build_registry(config_path: str | Path = "config/commands.yaml") -> CommandRegistry:
    data = load_yaml(config_path)
    registry = CommandRegistry()

    for item in data.get("commands", []):
        command = CommandDefinition(
            name=item["name"],
            aliases=item.get("aliases", []),
            description=item.get("description", ""),
            handler=resolve_handler(item["handler"]),
        )
        registry.register(command)

    return registry
