from __future__ import annotations

from dataclasses import dataclass,field
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any
from uuid import uuid4

def get_ist_time():
    return datetime.now(ZoneInfo("Asia/Kolkata"))

@dataclass(slots=True)
class MemoryMessage:
    """
    Internal representation of a conversation message.
    This model is used throughout the memory subsystem.
    """

    role:str
    content:str
    id:str = field(default_factory=lambda: str(uuid4()))
    metadata:dict[str,Any] = field(default_factory=dict)
    timestamp:datetime=field(default_factory=get_ist_time)

    def to_dict(self):
        return {
            "role":self.role,
            "content":self.content
        }
    
    @classmethod
    def from_dict(
        cls,message:dict[str,Any]
    ) -> 'MemoryMessage':
        """
        convvert an existing agentapi into MemoryMessage.
        """
        return cls(
            role = message['role'],
            content = message['content'],
            metadata = message.get('metadata',{}),

        )

@dataclass(slots=True)
class RetrievedMemory:
    """
    Represents a semantically retrieved memory.

    Returned by the VectorStore.
    """

    message: MemoryMessage
    similarity: float

@dataclass(slots=True)
class ConversationSummary:
    """
    Stores the compressed representation
    of older conversation history.
    """

    summary: str
    last_message_id: str | None = None

@dataclass(slots=True)
class MemoryContext:
    """
    Represents everything needed
    to construct an LLM prompt.
    """

    recent_messages: list[MemoryMessage]
    retrieved_memories: list[RetrievedMemory]
    summary: ConversationSummary | None = None