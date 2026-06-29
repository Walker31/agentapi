from __future__ import annotations

from abc import ABC, abstractmethod

from agentapi.core.memory.document import MemoryDocument


class BaseConversationStore(ABC):
    """
    Abstract interface for persistent conversation storage.

    A ConversationStore is responsible only for storing and retrieving
    chronological conversation history.

    Each store instance is bound to a single conversation.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the storage backend.

        Implementations should create any required tables,
        collections, or indexes if they do not already exist.
        """

    @abstractmethod
    async def create_conversation(self) -> None:
        """
        Create the current conversation if it does not exist.
        """

    @abstractmethod
    async def save_document(
        self,
        document: MemoryDocument,
    ) -> None:
        """
        Persist a single document.
        """

    @abstractmethod
    async def get_recent_documents(
        self,
        limit: int,
    ) -> list[MemoryDocument]:
        """
        Return the latest documents in chronological order.
        """

    @abstractmethod
    async def clear_conversation(self) -> None:
        """
        Delete all messages belonging to the current conversation.
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Close any open database connections.
        """