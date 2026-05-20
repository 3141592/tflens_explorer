"""Compare Service"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tflens_explorer.services.model_service import tokens_for_snapshot, logits_for_snapshot, cache_summary_for_snapshot
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
    prepend_bos: bool

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
            "prepend_bos": self.model.prepend_bos
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
    cache: list[int] = field(default_factory=list)

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

def snapshot_create(context: CommandContext, snapshot_name: str, layer: str) -> None:
    """Create a model comparison snapshot."""
    current_model = context.session.model
    prompt = context.session.current_prompt
    prepend_bos = context.session.prepend_bos

    model = Model(
        name=context.session.current_model_name,
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        num_ctx=2048,
        prepend_bos=context.session.prepend_bos
    )

    snapshot = Snapshot(
        name=snapshot_name,
        model=model,
        prompt=prompt,
        tokens=tokens_for_snapshot(current_model, prompt, prepend_bos),
        logits=logits_for_snapshot(current_model, prompt, prepend_bos),
        cache=cache_summary_for_snapshot(current_model, prompt, layer),
    )
    
    snapshot.save()
