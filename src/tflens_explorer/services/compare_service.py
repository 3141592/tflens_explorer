"""Compare Service"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tflens_explorer.services.model_service import tokens, logits, cache_run
from tflens_explorer.core.types import CommandContext

base_dir = Path(__file__).resolve().parents[3]
snapshot_path = base_dir / "snapshots"
snapshot_path.mkdir(parents=True, exist_ok=True)

@dataclass
class Model:
    name: str
    temperature: float
    top_k: int
    top_p: float
    num_ctx: int

    def save(self) -> None:
        path = snapshot_path / f"{self.name}.yaml"
        serial = dict(self.__dict__)                    # shallow copy
        # convert model to plain dict (choose attributes you need)
        serial["model"] = {
            "name": self.model.name,
            "temperature": self.model.temperature,
            "top_k": self.model.top_k,
            "top_p": self.model.top_p,
            "num_ctx": self.model.num_ctx,
        }
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(serial, f, sort_keys=False)

@dataclass
class Snapshot:
    name: str
    model: Model
    prompt: str
    tokens: list[int] = field(default_factory=list)
    logits: list[float] = field(default_factory=list)
    _cache: list[int] = field(default_factory=list)

    def save(self) -> None:
        """Save the snapshot to a YAML file."""
        try:
            path = snapshot_path / f"{self.name}.yaml"

            # Convert Model object
            serial = dict(self.__dict__)
            serial["model"] = asdict(self.model)

            with path.open("w", encoding="utf-8") as f:
                # dump a plain dict instead of the object to avoid yaml picking class internals
                yaml.safe_dump(serial, f, sort_keys=False)
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
    current_model = context.session.model
    prompt = context.session.current_prompt
    prepend_bos = context.session.prepend_bos

    model = Model(
        name=context.session.current_model_name,
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        num_ctx=2048
    )
    
    snapshot = Snapshot(
        name=snapshot_name,
        model=model,
        prompt=prompt,
        tokens=tokens(current_model, prompt, prepend_bos),
        logits=logits(current_model, prompt, prepend_bos),
    )
    
    snapshot.save()
