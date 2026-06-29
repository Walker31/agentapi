"""
Abstract embedding provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import Embedding


class BaseEmbeddingProvider(ABC):

    @abstractmethod
    async def embed(
        self,
        text: str,
    ) -> Embedding:
        """
        Generate an embedding.
        """

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[Embedding]:
        """
        Default batch implementation.
        """

        return [
            await self.embed(text)
            for text in texts
        ]