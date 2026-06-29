from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class VectorRecord:
    """
    Represents one vector stored inside a vector database.
    """
    id: str
    vector: list[float]
    metadata: dict[str, Any]

@dataclass(slots=True)
class SearchResult:
    """
    Represents a vector search result.
    """
    score: float
    record: VectorRecord