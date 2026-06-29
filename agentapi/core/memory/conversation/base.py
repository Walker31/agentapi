from __future__ import annotations

from abc import ABC, abstractmethod

from agentapi.core.memory.types import MemoryMessage


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
    async def save_message(
        self,
        message: MemoryMessage,
    ) -> None:
        """
        Persist a single message.
        """

    @abstractmethod
    async def get_recent_messages(
        self,
        limit: int,
    ) -> list[MemoryMessage]:
        """
        Return the latest messages in chronological order.
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