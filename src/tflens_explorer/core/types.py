"""Shared types used across the app."""

from dataclasses import dataclass


@dataclass
class CommandContext:
    raw: str
    args: list[str]
    session: object
    registry: object

    def __iter__(self):
        yield "args", self.args
        yield "session", self.session
        yield "registry", self.registry
