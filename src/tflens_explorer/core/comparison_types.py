from dataclasses import dataclass

@dataclass(frozen=True)
class HeadSimilarity:
    hook: str
    head: int
    cosine: float
    angle: float

