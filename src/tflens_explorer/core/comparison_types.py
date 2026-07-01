from dataclasses import dataclass

@dataclass(frozen=True)
class Column:
    title: str
    field_name: str
    width: int
    align: str = "<" # "<", ">", or "^"

@dataclass(frozen=True)
class Model:
    run: str
    name: str

@dataclass(frozen=True)
class Prompt:
    run: str
    diff_token_ids: str

@dataclass(frozen=True)
class Logits:
    run: str
    lenth: str
    top_1: str
    top_5: str

@dataclass(frozen=True)
class CacheActivationDifferencesRow:
    run: str
    hook: str
    shape: str
    min: float
    max: float
    mean: float
    mean_abs_diff: float
    cos_sim: float

@dataclass(frozen=True)
class HeadSimilarity:
    hook: str
    head: int
    cosine: float
    angle: float

AnugularSimilarityColumns = [
    Column("Layer", "layer", 36),
    Column("Head", "head", 5, ">"),
    Column("Angle", "angle", 8, ">")
]

@dataclass(frozen=True)
class AngularSimilarityRow:
    layer: str
    head: int
    angle: float
