""" Compare Service """

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from tflens_explorer.services.model_service import tokens
from tflens_explorer.services.model_service import logits
from tflens_explorer.services.model_service import cache_run
from tflens_explorer.core.types import CommandContext

base_dir = Path(__file__).resolve().parents[3]
snapshot_path = base_dir / "snapshots"
snapshot_path.mkdir(parents=True, exist_ok=True)

class Snapshot:
    name: str
    model: str
    prompt: str
    tokens: list[int] = field(default_factory=list)
    logits: list[float] = field(default_factory=list)
    _cache: list[int] = field(default_factory=list)

    def save(self) -> None:
        path = snapshot_path / f"{self.name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            # dump a plain dict instead of the object to avoid yaml picking class internals
            yaml.safe_dump(self.__dict__, f, sort_keys=False)        

class Compare:
    def __init__(self, name) -> None:
        self.snaphot = Snapshot()
        self.snapshot.name = name

def compare(context: CommandContext, snapshot_name: str):
    model = context.session.model
    prompt = context.session.current_prompt
    prepend_bos = context.session.prepend_bos

    snapshot = Snapshot()
    snapshot.name = snapshot_name
    snapshot.model = context.session.current_model_name
    snapshot.prompt = prompt

    snapshot.tokens = tokens(model, prompt, prepend_bos)
    snapshot.logits = logits(model, prompt, prepend_bos)
    #snapshot.cache = None
    #snapshot.cache = cache_run(model, prompt)
    #breakpoint() 
    snapshot.save()
    
    return