import os
import yaml
import torch
import re
import datetime
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

@dataclass
class SnapshotMetadata:
    name: str | None = None
    creation_date: str | None = None

@dataclass
class TokenSummary:
    index: int
    token_id: int
    token: str

@dataclass
class TokenCollection:
    shape: str
    values: list[TokenSummary]

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

@dataclass
class Snapshot:
    metadata: SnapshotMetadata | None = None
    model: Model | None = None
    prompt: str | None = None
    token_shape: str | None = None
    tokens: list[TokenSummary] = field(default_factory=list)
    logits: list[float] = field(default_factory=list)
    cache: list[int] = field(default_factory=list)

    def save(self) -> None:
        """Save the snapshot to a YAML file."""
        try:
            path = snapshot_yaml_path(self.metadata.name)
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("w", encoding="utf-8") as f:
                # dump a plain dict instead of the object to avoid yaml picking class internals
                yaml.safe_dump(asdict(self), f, sort_keys=False)
            print(f"Snapshot saved to {path}")
        except Exception as e:
            print(f"Error saving snapshot: {str(e)}")
            raise

    @classmethod
    def load(cls, name: str) -> "Snapshot":
        path = SNAPSHOT_PATH / name / "snapshot.yaml"

        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            token_values = data.get("tokens", [])

            tokens = [
                TokenSummary(**token_data)
                for token_data in token_values
            ]

            cache = [
                CacheSummary(**cache_data)
                for cache_data in data["cache"]
            ]

            return cls(
                metadata=SnapshotMetadata(**data["metadata"]),
                model=Model(**data["model"]) if data.get("model") else None,
                prompt=data.get("prompt"),
                token_shape=data.get('token_shape'),
                tokens=tokens,
                logits=data.get("logits", []),
                cache=cache,
            )
        except Exception as error:
            print(error)
            breakpoint()
            exit

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

