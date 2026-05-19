"""Application session state."""

from dataclasses import dataclass, field, fields


@dataclass
class AppSession:
    running: bool = True
    current_model_name: str | None = None
    current_prompt: str = ""
    model: object | None = None
    cache: object | None = None
    logits: object | None = None
    tokens: object | None = None
    scratch: dict = field(default_factory=dict)
    last_output: str = ""
    prepend_bos: bool = True

    def __iter__(self):
        for f in fields(self):
            yield f.name, getattr(self, f.name)

    def clear_runtime_state(self, keep_model: bool = True) -> None:
        self.logits = None
        self.cache = None
        self.tokens = None
        self.scratch.clear()
        self.last_output = ""

        if not keep_model:
            self.model = None
            self.prepend_bos = None
            self.current_model_name = ""

    def clear_session(self) -> None:
        self.current_prompt = ""
        self.logits = None
        self.cache = None
        self.tokens = None
        self.scratch.clear()
        self.last_output = ""
        self.model = None
        self.prepend_bos = None
        self.current_model_name = ""
