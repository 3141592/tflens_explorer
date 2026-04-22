"""Parse raw user input and dispatch commands."""

import shlex

from tflens_explorer.core.types import CommandContext


class CommandDispatcher:
    def __init__(self, registry, session) -> None:
        self.registry = registry
        self.session = session

    def dispatch(self, raw: str) -> None:
        parts = shlex.split(raw)
        command_name = parts[0]
        args = parts[1:]

        command = self.registry.get(command_name)
        if command is None:
            print(f"Unknown command: {command_name}")
            return

        context = CommandContext(
            raw=raw,
            args=args,
            session=self.session,
            registry=self.registry,
        )
        command.handler(context)
