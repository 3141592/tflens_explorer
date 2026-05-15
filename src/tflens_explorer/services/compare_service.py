""" Compare Service """

import os
import yaml
from pathlib import Path

base_dir = Path(__file__).resolve().parents[3]
snapshot_path = base_dir / "snapshots"

class Snapshot:
    name: str
    model: str
    prompt: str
    tokens: list[int]
    logits: list[int]
    cache: list[int]

    def save(obj):
        with open("obj.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, sort_keys=False)
    

class Compare:
    def __init__(self, name) -> None:
        self.snaphot = Snapshot()
        self.snapshot.name = name

def compare(model):
    return