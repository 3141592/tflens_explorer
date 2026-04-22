"""Application session state."""

from dataclasses import dataclass, field


@dataclass
class AppSession:
    running: bool = True
    current_model_name: str | None = None
    current_prompt: str = ""
    model: object | None = None
    cache: object | None = None
    scratch: dict = field(default_factory=dict)
    last_output: str = ""
