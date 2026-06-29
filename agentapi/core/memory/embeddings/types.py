from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class Embedding:
    """
    Represents a generated embedding.
    """
    vector: list[float]
    model: str
    dimensions: int
    metadata: dict[str, Any] | None = None