from __future__ import annotations

from abc import ABC, abstractmethod

class BaseEmbeddingProvider(ABC):
    """
    Generates vector embeddings for text.

    HybridMemory depends only on this interface,
    allowing any embedding model to be plugged in.
    """

    @abstractmethod
    async def embed(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a single piece of text.
        """

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        """
    
