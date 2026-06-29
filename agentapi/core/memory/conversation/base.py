from __future__ import annotations
from abc import ABC,abstractmethod
from typing import Any

class BaseConversationStore(ABC):
    """
    Abstract interface for persistent conversation storage.

    A conversation store is responsible only for storing and retrieving chronological chat history

    It MUST NOT:
        - Generate embeddings
        - perform vector search
        - Build prompts
        - Call LLMs
    """

    @abstractmethod
    async def create_conversation(
        self,conversation_id:str,
    ) -> None:
        """Create a conversation if it doesn't already exist."""
        pass

    @abstractmethod
    async def save_messages(
        self,conversation_id:str,
        message: dict[str,Any],
    ) -> None:
        """ Persistent a single message"""

    @abstractmethod
    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent messages in chronological order.
        """

    @abstractmethod
    async def clear_conversation(
        self,
        conversation_id: str,
    ) -> None:
        """Delete every message in a conversation."""
