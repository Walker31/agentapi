"""
Hybrid memory backend combining conversation history and vector search.
"""

from __future__ import annotations

import logging
from typing import Any

from agentapi.agent.memory import MemoryBackend, create_conversation_id
from agentapi.core.memory.document import MemoryDocument
from agentapi.core.memory.vector.types import SearchResult, VectorRecord

from .conversation.base import BaseConversationStore
from .embeddings.base import BaseEmbeddingProvider
from .retry import EmbeddingRetryQueue
from .vector.base import BaseVectorStore

logger = logging.getLogger(__name__)


class HybridMemory(MemoryBackend):
    """
    Hybrid memory backend combining conversation history and vector search.

    Pipeline:
        aadd()  →  save to ConversationStore  →  embed  →  upsert to VectorStore
        build_messages()  →  load recent  →  retrieve semantic  →  merge context
    """

    def __init__(
        self,
        *,
        embedding_provider: BaseEmbeddingProvider | None = None,
        conversation_store: BaseConversationStore | None = None,
        vector_store: BaseVectorStore | None = None,
        conversation_id: str | None = None,
        recent_window: int = 20,
        retrieval_top_k: int = 5,
    ) -> None:

        self._embedding_provider = embedding_provider
        self._conversation_store = conversation_store
        self._vector_store = vector_store
        self.conversation_id = conversation_id or create_conversation_id()
        self._recent_window = recent_window
        self._retrieval_top_k = retrieval_top_k
        self._retry_queue = EmbeddingRetryQueue()

    # ------------------------------------------------------------------
    # Synchronous interface (not supported)
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the current conversation messages.

        Note: For HybridMemory, use the async build_messages() or
        get_recent_documents() method instead.
        """
        raise NotImplementedError(
            "HybridMemory is asynchronous. Use async build_messages() or get_recent_documents() instead."
        )

    def add(self, message: dict[str, Any]) -> None:
        """Synchronous add is not supported on HybridMemory. Use aadd() instead."""
        raise NotImplementedError(
            "HybridMemory is asynchronous. Use async aadd() instead."
        )

    # ------------------------------------------------------------------
    # Write pipeline
    # ------------------------------------------------------------------

    async def aadd(
        self,
        message: dict[str, Any] | MemoryDocument,
    ) -> None:
        """Persist a conversation message and index its vector."""

        from datetime import datetime, timezone
        from uuid import uuid4

        if isinstance(message, MemoryDocument):
            doc = message
        elif isinstance(message, dict):
            doc_id = message.get("id") or str(uuid4())
            doc_metadata = message.get("metadata") or {}
            doc_timestamp = message.get("timestamp")

            if isinstance(doc_timestamp, str):
                doc_timestamp = datetime.fromisoformat(doc_timestamp)
            elif not isinstance(doc_timestamp, datetime):
                doc_timestamp = datetime.now(timezone.utc)

            doc = MemoryDocument(
                id=doc_id,
                conversation_id=self.conversation_id,
                role=message["role"],
                content=message["content"],
                metadata=doc_metadata,
                timestamp=doc_timestamp,
            )
        else:
            raise TypeError("message must be a dict or MemoryDocument")

        await self._persist_message(doc)

    async def _persist_message(
        self,
        document: MemoryDocument,
    ) -> None:
        """
        Save document to the conversation store, then attempt
        embedding + vector upsert. On failure, queue for retry.
        """

        # Step 1: Always save to conversation store (source of truth)
        if self._conversation_store is not None:
            await self._conversation_store.save_document(document)

        # Step 2: Embed and upsert (if providers are available)
        if self._embedding_provider is None or self._vector_store is None:
            return

        try:
            embedding = await self._embedding_provider.embed(
                document.content,
            )

            await self._vector_store.upsert(
                VectorRecord(
                    id=document.id,
                    vector=embedding.vector,
                    metadata={
                        "conversation_id": document.conversation_id,
                        "role": document.role,
                        "content": document.content,
                        "timestamp": document.timestamp.isoformat(),
                    },
                )
            )

        except Exception as exc:
            await self._retry_queue.push(
                document,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Read pipeline
    # ------------------------------------------------------------------

    async def build_messages(
        self,
        current_query: str,
    ) -> list[dict[str, Any]]:
        """
        Build the full conversation context.

        Pipeline:
            _load_recent_messages  →  _retrieve_semantic_memories  →  _merge_context
        """

        if self._conversation_store is None:
            return []

        recent = await self._load_recent_messages()

        if self._embedding_provider is None or self._vector_store is None:
            return [
                {"role": d.role, "content": d.content}
                for d in recent
            ]

        semantic = await self._retrieve_semantic_memories(current_query)

        return self._merge_context(recent, semantic)

    async def _load_recent_messages(
        self,
    ) -> list[MemoryDocument]:
        """Fetch the most recent documents from the conversation store."""

        return await self._conversation_store.get_recent_documents(
            self._recent_window,
        )

    async def _retrieve_semantic_memories(
        self,
        query: str,
    ) -> list[SearchResult]:
        """Embed the query and search the vector store for relevant memories."""

        embedding = await self._embedding_provider.embed(query)

        return await self._vector_store.search(
            embedding.vector,
            top_k=self._retrieval_top_k,
            filters={
                "conversation_id": self.conversation_id,
            },
        )

    def _merge_context(
        self,
        recent: list[MemoryDocument],
        semantic: list[SearchResult],
    ) -> list[dict[str, Any]]:
        """
        Merge recent messages with semantic search results.

        Algorithm (O(n)):
            1. Collect IDs of recent messages into a set
            2. Filter semantic results to only those NOT in recent
            3. Sort semantic-only results by timestamp ascending
            4. Prepend semantic-only before recent messages
        """

        recent_ids = {doc.id for doc in recent}

        semantic_only = [
            sr for sr in semantic
            if sr.record.id not in recent_ids
        ]

        semantic_only.sort(
            key=lambda sr: sr.record.metadata.get("timestamp", ""),
        )

        merged: list[dict[str, Any]] = []

        # Semantic-only memories first (older relevant context)
        for sr in semantic_only:
            merged.append({
                "role": sr.record.metadata.get("role", "user"),
                "content": sr.record.metadata.get("content", ""),
            })

        # Recent messages in chronological order
        for doc in recent:
            merged.append({
                "role": doc.role,
                "content": doc.content,
            })

        return merged

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Reset the conversation store and vector store."""

        if self._conversation_store is not None:
            await self._conversation_store.clear_conversation()

        if self._vector_store is not None:
            await self._vector_store.delete(
                filters={"conversation_id": self.conversation_id},
            )

    async def close(self) -> None:
        """Close all backend connections."""

        if self._conversation_store is not None:
            await self._conversation_store.close()

        if self._vector_store is not None:
            await self._vector_store.close()