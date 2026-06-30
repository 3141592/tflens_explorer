from dataclasses import dataclass

@dataclass(frozen=True)
class Column:
    title: str
    field_name: str
    width: int
    align: str = "<"

@dataclass(frozen=True)
class HeadSimilarity:
    hook: str
    head: int
    cosine: float
    angle: float

@dataclass(frozen=True)
class CacheActivationDifferences:
    run: str
    hook: str
    shape: str
    min: float
    max: float
    mean: float
    mean_abs_diff: float
    cos_sim: float
