"""Command registry built from YAML config."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tflens_explorer.cli import core_handlers
from tflens_explorer.utils.yaml_loader import load_yaml

_HANDLER_CACHE = {}

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
    if handler_name in _HANDLER_CACHE:
        return _HANDLER_CACHE[handler_name]

    try:
        module_path, function_name = handler_name.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid handler path: {handler_name}"
        ) from exc

    full_module_path = f"tflens_explorer.{module_path}"

    module = importlib.import_module(full_module_path)

    try:
        handler = getattr(module, function_name)
    except AttributeError as exc:
        raise ValueError(
            f"Handler not found: {function_name} in {full_module_path}"
        ) from exc

    _HANDLER_CACHE[handler_name] = handler
    return handler


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
