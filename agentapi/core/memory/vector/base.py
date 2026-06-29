from __future__ import annotations

from abc import ABC,abstractmethod
from typing import Any

class BaseVectorStore(ABC):
    """
    Stores and retrieves semantic vectors.
    """

    @abstractmethod
    async def upsert(
        self,
        conversation_id:str,
        message_id:str,
        embeddings:list[float],
        metadata:dict[str,Any]
    ) -> None:
        """Insert or update a vector."""
    
    @abstractmethod
    async def search(
        self,
        conversation_id:str,
        embeddings:list[float],
        top_k:int
    ) -> list[dict[str,Any]]:
        """Search for similar messages"""

    @abstractmethod
    async def delete(
        self,conversation_id:str
    ) -> None:
        """Delete all vectors for a conversation"""
    
    @abstractmethod
    async def close(self) -> None:
        """Release any resources"""