import os
import yaml
import torch
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict

BASE_DIR = Path(__file__).resolve().parents[3]
SNAPSHOT_PATH = BASE_DIR / "snapshots"
SNAPSHOT_PATH.mkdir(parents=True, exist_ok=True)

@dataclass
class Model:
    name: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    num_ctx: int | None = None
    prepend_bos: bool | None = None
    layers: int | None = None
    heads: int | None = None
    vocabulary: int | None = None    

    def save(self) -> None:
        path = SNAPSHOT_PATH / f"{self.name}.yaml"
        serial = dict(self.__dict__)                    # shallow copy
        # convert model to plain dict (choose attributes you need)
        serial["model"] = {
            "name": self.model.name,
            "temperature": self.model.temperature,
            "top_k": self.model.top_k,
            "top_p": self.model.top_p,
            "num_ctx": self.model.num_ctx,
            "prepend_bos": self.model.prepend_bos,
            "layers": self.model.layers,
            "heads": self.model.heads,
            "vocabulary": self.model.vocabulary,
        }
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(serial, f, sort_keys=False)

@dataclass
class Snapshot:
    name: str | None = None
    model: Model | None = None
    prompt: str | None = None
    tokens: list[int] = field(default_factory=list)
    logits: list[float] = field(default_factory=list)
    cache: list[int] = field(default_factory=list)

    def save(self) -> None:
        """Save the snapshot to a YAML file."""
        try:
            path = snapshot_yaml_path(self.name)
            path.parent.mkdir(parents=True, exist_ok=True)

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

    def load(self) -> None:
        """Load the snapshot to a Snapshot object."""
        try:
            path = snapshot_yaml_path(self.name)

            with open(path, 'r') as file:
                data = yaml.safe_load(file)

            self.name = data['name']
            self.prompt = data['prompt']
            # Model
            self.model = Model(**data["model"])
            
            # Tokens
            self.tokens = []
            for item in data['tokens']:
                self.tokens.append(item)

            # Logits
            self.logits = []
            for item in data['logits']:
                self.logits.append(item)

            # Cache
            if isinstance(data["cache"], list):
                self.cache = [CacheSummary(**item) for item in data["cache"]]
            elif isinstance(data["cache"], dict):
                self.cache = [CacheSummary(**data["cache"])]
            else:
                print(f"Error loading cache. Check the snapshot cache.")
                print()
                return

        except Exception as error:
            print(f"Exception: {error}")

@dataclass
class CacheSummary:
    hook: str
    shape: list[int]
    dtype: str
    device: str
    minimum: float
    maximum: float
    mean: float | str
    std: float | str
    numel: int

class Compare:
    def __init__(self, name: str) -> None:
        self.snapshot = Snapshot(name=name)
        # Additional initialization could be added here

def snapshot_dir(snapshot_name: str) -> Path:
    return SNAPSHOT_PATH / snapshot_name

def snapshot_yaml_path(snapshot_name: str) -> Path:
    return snapshot_dir(snapshot_name) / "snapshot.yaml"

def verify_snapshot(snapshot_name):
    path = snapshot_yaml_path(snapshot_name)

    if path.exists():
        return True
    else:
        return False

