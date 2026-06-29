from __future__ import annotations

from typing import Any
import pytest
from datetime import datetime, timezone

from agentapi.core.memory.hybrid import HybridMemory
from agentapi.core.memory.conversation.base import BaseConversationStore
from agentapi.core.memory.vector.base import BaseVectorStore
from agentapi.core.memory.embeddings.base import BaseEmbeddingProvider
from agentapi.core.memory.embeddings.types import Embedding
from agentapi.core.memory.vector.types import VectorRecord, SearchResult
from agentapi.core.memory.document import MemoryDocument


# ------------------------------------------------------------------
# Mock backends
# ------------------------------------------------------------------


class MockConversationStore(BaseConversationStore):
    def __init__(self) -> None:
        self.documents: list[MemoryDocument] = []
        self.initialized = False
        self.conversation_created = False
        self.cleared = False
        self.closed = False

    async def initialize(self) -> None:
        self.initialized = True

    async def create_conversation(self) -> None:
        self.conversation_created = True

    async def save_document(self, document: MemoryDocument) -> None:
        self.documents.append(document)

    async def get_recent_documents(self, limit: int) -> list[MemoryDocument]:
        return self.documents[-limit:]

    async def clear_conversation(self) -> None:
        self.documents.clear()
        self.cleared = True

    async def close(self) -> None:
        self.closed = True


class MockEmbeddingProvider(BaseEmbeddingProvider):
    async def embed(self, text: str) -> Embedding:
        return Embedding(
            vector=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3,
        )


class FailingEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider that always raises."""

    def __init__(self) -> None:
        self.call_count = 0

    async def embed(self, text: str) -> Embedding:
        self.call_count += 1
        raise RuntimeError("Gemini temporarily unavailable")


class FailThenSucceedEmbeddingProvider(BaseEmbeddingProvider):
    """Fails on the first call, succeeds on subsequent calls."""

    def __init__(self) -> None:
        self.call_count = 0

    async def embed(self, text: str) -> Embedding:
        self.call_count += 1
        if self.call_count <= 1:
            raise RuntimeError("Transient failure")
        return Embedding(
            vector=[0.4, 0.5, 0.6],
            model="test-model",
            dimensions=3,
        )


class MockVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.searched = False
        self.deleted_filters: dict[str, Any] | None = None
        self.closed = False

    async def initialize(self) -> None:
        pass

    async def upsert(self, record: VectorRecord) -> None:
        self.records[record.id] = record

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.searched = True
        return [
            SearchResult(score=1.0, record=r)
            for r in self.records.values()
        ]

    async def delete(self, *, filters: dict[str, Any]) -> None:
        self.deleted_filters = filters
        self.records.clear()

    async def close(self) -> None:
        self.closed = True


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ------------------------------------------------------------------
# Existing tests (updated for new merge behavior)
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_hybrid_memory_without_vectors():
    """HybridMemory should work fine with just a conversation store."""
    conv_store = MockConversationStore()
    memory = HybridMemory(
        conversation_store=conv_store,
        conversation_id="test-conv",
    )

    # 1. Test aadd() saves document but skips embedding/vector store
    await memory.aadd({"role": "user", "content": "Hello World"})
    assert len(conv_store.documents) == 1
    assert conv_store.documents[0].content == "Hello World"

    # 2. Test build_messages() retrieves from store without embedding call
    msgs = await memory.build_messages("test query")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello World"


@pytest.mark.anyio
async def test_hybrid_memory_with_vectors():
    """HybridMemory should embed and upsert if all components are present."""
    conv_store = MockConversationStore()
    vector_store = MockVectorStore()
    embed_provider = MockEmbeddingProvider()

    memory = HybridMemory(
        conversation_store=conv_store,
        vector_store=vector_store,
        embedding_provider=embed_provider,
        conversation_id="test-conv",
    )

    # 1. Test aadd() saves and embeds/upserts
    await memory.aadd({"role": "user", "content": "Hello Hybrid"})
    assert len(conv_store.documents) == 1
    assert len(vector_store.records) == 1

    # Verify the document metadata in vector store includes content
    doc_id = conv_store.documents[0].id
    assert doc_id in vector_store.records
    record = vector_store.records[doc_id]
    assert record.vector == [0.1, 0.2, 0.3]
    assert record.metadata["role"] == "user"
    assert record.metadata["content"] == "Hello Hybrid"
    assert "timestamp" in record.metadata

    # 2. Test build_messages() performs semantic search and returns messages
    msgs = await memory.build_messages("query text")
    assert vector_store.searched is True
    # Same message in both recent and semantic → no duplication
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Hello Hybrid"


@pytest.mark.anyio
async def test_hybrid_memory_reset_and_close():
    """HybridMemory reset and close should clean up all stores."""
    conv_store = MockConversationStore()
    vector_store = MockVectorStore()

    memory = HybridMemory(
        conversation_store=conv_store,
        vector_store=vector_store,
        conversation_id="test-conv",
    )

    # Test reset
    await memory.reset()
    assert conv_store.cleared is True
    assert vector_store.deleted_filters == {"conversation_id": "test-conv"}

    # Test close
    await memory.close()
    assert conv_store.closed is True
    assert vector_store.closed is True


def test_hybrid_memory_sync_methods_raise():
    """Synchronous methods on HybridMemory should raise NotImplementedError."""
    memory = HybridMemory(conversation_id="test-conv")

    with pytest.raises(NotImplementedError):
        _ = memory.messages

    with pytest.raises(NotImplementedError):
        memory.add({"role": "user", "content": "Hello"})


# ------------------------------------------------------------------
# New: Merge deduplication
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_merge_deduplication():
    """
    Messages present in both recent and semantic results
    should appear only once, in the recent section.
    """
    conv_store = MockConversationStore()
    embed_provider = MockEmbeddingProvider()

    # Custom vector store that returns a mix of recent + old results
    class MergeVectorStore(MockVectorStore):
        async def search(
            self,
            vector: list[float],
            *,
            top_k: int = 5,
            filters: dict[str, Any] | None = None,
        ) -> list[SearchResult]:
            self.searched = True
            return [
                SearchResult(score=1.0, record=r)
                for r in self.records.values()
            ] + [
                # Old semantic-only memory
                SearchResult(
                    score=0.8,
                    record=VectorRecord(
                        id="old-memory",
                        vector=[0.7, 0.8, 0.9],
                        metadata={
                            "conversation_id": "test-conv",
                            "role": "user",
                            "content": "My favourite IDE is VSCode",
                            "timestamp": "2024-01-01T00:00:00+00:00",
                        },
                    ),
                ),
            ]

    vector_store = MergeVectorStore()

    memory = HybridMemory(
        conversation_store=conv_store,
        vector_store=vector_store,
        embedding_provider=embed_provider,
        conversation_id="test-conv",
    )

    # Add recent messages
    await memory.aadd({"role": "user", "content": "Hello"})
    await memory.aadd({"role": "user", "content": "I own a MacBook"})
    await memory.aadd({"role": "user", "content": "Thanks"})

    msgs = await memory.build_messages("What laptop do I own?")

    # Should be: [VSCode (semantic-only)] + [Hello, MacBook, Thanks (recent)]
    assert len(msgs) == 4

    # Semantic-only memory comes first
    assert msgs[0]["content"] == "My favourite IDE is VSCode"

    # Recent messages follow in chronological order
    assert msgs[1]["content"] == "Hello"
    assert msgs[2]["content"] == "I own a MacBook"
    assert msgs[3]["content"] == "Thanks"


# ------------------------------------------------------------------
# New: Retry queue on embedding failure
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_retry_queue_on_embedding_failure():
    """
    When embedding fails, the message should be saved to PostgreSQL
    and queued for retry. The retry should succeed later.
    """
    conv_store = MockConversationStore()
    vector_store = MockVectorStore()
    failing_provider = FailingEmbeddingProvider()

    memory = HybridMemory(
        conversation_store=conv_store,
        vector_store=vector_store,
        embedding_provider=failing_provider,
        conversation_id="test-conv",
    )

    # aadd should NOT raise even though embedding fails
    await memory.aadd({"role": "user", "content": "Test message"})

    # PostgreSQL has the message (source of truth)
    assert len(conv_store.documents) == 1

    # Qdrant does NOT have it
    assert len(vector_store.records) == 0

    # Retry queue has the item
    assert memory._retry_queue.pending_count == 1

    # Now swap in a working provider and retry
    working_provider = MockEmbeddingProvider()
    processed = await memory._retry_queue.process_pending(
        working_provider,
        vector_store,
    )

    assert processed == 1
    assert memory._retry_queue.pending_count == 0
    assert len(vector_store.records) == 1


@pytest.mark.anyio
async def test_retry_queue_max_attempts():
    """
    Items should be dropped after exceeding max retry attempts.
    """
    conv_store = MockConversationStore()
    vector_store = MockVectorStore()
    failing_provider = FailingEmbeddingProvider()

    memory = HybridMemory(
        conversation_store=conv_store,
        vector_store=vector_store,
        embedding_provider=failing_provider,
        conversation_id="test-conv",
    )

    await memory.aadd({"role": "user", "content": "Will fail"})
    assert memory._retry_queue.pending_count == 1

    # Retry with a provider that keeps failing (max_attempts defaults to 3)
    # Attempt 1 was the initial push, attempts 2 and 3 via process_pending
    await memory._retry_queue.process_pending(failing_provider, vector_store)
    assert memory._retry_queue.pending_count == 1  # attempt 2, still under limit

    await memory._retry_queue.process_pending(failing_provider, vector_store)
    assert memory._retry_queue.pending_count == 0  # attempt 3, dropped
    assert len(vector_store.records) == 0
