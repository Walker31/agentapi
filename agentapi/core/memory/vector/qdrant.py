"""
Qdrant vector store implementation.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from .base import BaseVectorStore
from .types import SearchResult, VectorRecord


def _build_filter(filters: dict) -> models.Filter:
    """Convert a flat dict of key-value pairs into a Qdrant Filter."""

    return models.Filter(
        must=[
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in filters.items()
        ],
    )


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant-backed implementation of BaseVectorStore.

    Responsibilities:
        - Connection management
        - Collection initialization
        - Vector CRUD operations
        - Similarity search
    """

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "agentapi-memory",
        vector_size: int = 768,
    ) -> None:

        self._url = url
        self._api_key = api_key
        self._collection_name = collection_name
        self._vector_size = vector_size

        self._client: AsyncQdrantClient | None = None
        self._initialized = False

    async def _get_client(self) -> AsyncQdrantClient:
        """
        Lazily create the Qdrant client.
        """

        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self._url,
                api_key=self._api_key,
            )

        return self._client

    async def initialize(self) -> None:
        """
        Initialize the vector store.

        Creates the collection if it does not already exist.
        """

        if self._initialized:
            return

        client = await self._get_client()

        try:
            await client.get_collection(
                collection_name=self._collection_name,
            )

        except UnexpectedResponse:

            await client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

        self._initialized = True

    async def upsert(
        self,
        record: VectorRecord,
    ) -> None:
        """
        Insert or update a single vector point.
        """

        await self.initialize()

        client = await self._get_client()

        await client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.metadata,
                ),
            ],
        )

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Perform cosine similarity search with optional metadata filters.
        """

        await self.initialize()

        client = await self._get_client()

        query_filter = _build_filter(filters) if filters else None

        results = await client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            SearchResult(
                score=point.score,
                record=VectorRecord(
                    id=str(point.id),
                    vector=point.vector or [],
                    metadata=point.payload or {},
                ),
            )
            for point in results.points
        ]

    async def delete(
        self,
        *,
        filters: dict,
    ) -> None:
        """
        Delete all points matching the given metadata filters.
        """

        await self.initialize()

        client = await self._get_client()

        await client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=_build_filter(filters),
            ),
        )

    async def close(self) -> None:
        """
        Close the Qdrant client.
        """

        if self._client is None:
            return

        await self._client.close()
        self._client = None