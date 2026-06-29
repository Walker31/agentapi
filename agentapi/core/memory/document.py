from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass(slots=True)
class MemoryDocument:
    """
    Represents a conversation document throughout the memory subsystem.
    """
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: dict[str, Any]
    timestamp: datetime
