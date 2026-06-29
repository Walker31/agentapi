"""
Abstract vector database interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import SearchResult, VectorRecord


class BaseVectorStore(ABC):
    """
    Abstract interface for vector storage backends.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the vector database.
        """

    @abstractmethod
    async def upsert(
        self,
        record: VectorRecord,
    ) -> None:
        """
        Insert or update a vector.
        """

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Perform semantic similarity search.
        """

    @abstractmethod
    async def delete(
        self,
        *,
        filters: dict[str, Any],
    ) -> None:
        """
        Delete vectors matching filters.
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Close any open connections.
        """