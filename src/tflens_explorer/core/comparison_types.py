from dataclasses import dataclass

@dataclass(frozen=True)
class Column:
    title: str
    field_name: str
    width: int
    align: str = "<" # "<", ">", or "^"
    precision: int | None = None

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

CacheActivationDifferencesColumns = [
    Column("Run", "run", 4),
    Column("Layer", "layer", 35, "<"),
    Column("Shape", "shape", 15, ">"),
    Column("Min", "min", 12, ">", 4),
    Column("Max", "max", 12, ">", 4),
    Column("Mean", "mean", 12, ">"),
    Column("Mean Abs Diff", "mean_abs_diff", 16, ">"),
]

@dataclass(frozen=True)
class CacheActivationDifferencesRow:
    run: str
    layer: str
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
    Column("Head", "head", 6, ">"),
    Column("Angle", "angle", 10, ">", 2)
]

@dataclass(frozen=True)
class AngularSimilarityRow:
    layer: str
    head: int
    angle: float
