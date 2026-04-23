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
        "builtin.prompt_set": builtins.handle_prompt_set,
        "builtin.prompt_show": builtins.handle_prompt_show,
        "builtin.prompt_run": builtins.handle_prompt_run,
        "builtin.model_list": builtins.handle_model_list,
        "builtin.model_load": builtins.handle_model_load,
        "builtin.model_info": builtins.handle_model_info,
        "builtin.tokens": builtins.handle_tokens,
        "builtin.logits": builtins.handle_logits,
        "builtin.cache_run": builtins.handle_cache_run,
        "builtin.cache_show": builtins.handle_cache_show,
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
