"""Compare Service"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from tflens_explorer.services.model_service import tokens, logits, cache_run
from tflens_explorer.core.types import CommandContext

base_dir = Path(__file__).resolve().parents[3]
snapshot_path = base_dir / "snapshots"
snapshot_path.mkdir(parents=True, exist_ok=True)

@dataclass
class Snapshot:
    name: str
    model: str
    prompt: str
    tokens: list[int] = field(default_factory=list)
    logits: list[float] = field(default_factory=list)
    cache: list[int] = field(default_factory=list)

    def save(self) -> None:
        """Save the snapshot to a YAML file."""
        try:
            path = snapshot_path / f"{self.name}.yaml"
            with path.open("w", encoding="utf-8") as f:
                # dump a plain dict instead of the object to avoid yaml picking class internals
                yaml.safe_dump(self.__dict__, f, sort_keys=False)
            print(f"Snapshot saved to {path}")
        except Exception as e:
            print(f"Error saving snapshot: {str(e)}")
            raise

class Compare:
    def __init__(self, name: str) -> None:
        self.snapshot = Snapshot(name=name)
        # Additional initialization could be added here

def compare(context: CommandContext, snapshot_name: str) -> None:
    """Create a model comparison snapshot."""
    model = context.session.model
    prompt = context.session.current_prompt
    prepend_bos = context.session.prepend_bos
    
    snapshot = Snapshot(
        name=snapshot_name,
        model=context.session.current_model_name,
        prompt=prompt,
        tokens=tokens(model, prompt, prepend_bos),
        logits=logits(model, prompt, prepend_bos),
        cache=cache_run(model, prompt)
    )
    
    snapshot.save()
