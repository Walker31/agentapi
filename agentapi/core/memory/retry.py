"""
In-memory retry queue for failed embedding operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentapi.core.memory.document import MemoryDocument
from agentapi.core.memory.embeddings.base import BaseEmbeddingProvider
from agentapi.core.memory.vector.base import BaseVectorStore
from agentapi.core.memory.vector.types import VectorRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetryItem:
    """
    A single document whose embedding failed and needs retrying.
    """

    document: MemoryDocument
    attempts: int = 0
    last_error: str = ""


class EmbeddingRetryQueue:
    """
    In-memory queue for documents that failed embedding or vector upsert.

    PostgreSQL is the source of truth — the message is already saved.
    This queue ensures the corresponding vector eventually gets indexed.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
    ) -> None:

        self._queue: list[RetryItem] = []
        self._max_attempts = max_attempts

    @property
    def pending_count(self) -> int:
        """Return the number of items waiting for retry."""

        return len(self._queue)

    async def push(
        self,
        document: MemoryDocument,
        error: str = "",
    ) -> None:
        """Add a failed document to the retry queue."""

        self._queue.append(
            RetryItem(
                document=document,
                attempts=1,
                last_error=error,
            )
        )

        logger.warning(
            "Queued document %s for embedding retry: %s",
            document.id,
            error,
        )

    async def process_pending(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ) -> int:
        """
        Retry all pending items.

        Returns the number of successfully processed items.
        """

        if not self._queue:
            return 0

        succeeded = 0
        remaining: list[RetryItem] = []

        for item in self._queue:
            try:
                embedding = await embedding_provider.embed(
                    item.document.content,
                )

                await vector_store.upsert(
                    VectorRecord(
                        id=item.document.id,
                        vector=embedding.vector,
                        metadata={
                            "conversation_id": item.document.conversation_id,
                            "role": item.document.role,
                            "content": item.document.content,
                            "timestamp": item.document.timestamp.isoformat(),
                        },
                    )
                )

                succeeded += 1

                logger.info(
                    "Retry succeeded for document %s",
                    item.document.id,
                )

            except Exception as exc:
                item.attempts += 1
                item.last_error = str(exc)

                if item.attempts < self._max_attempts:
                    remaining.append(item)
                    logger.warning(
                        "Retry %d/%d failed for document %s: %s",
                        item.attempts,
                        self._max_attempts,
                        item.document.id,
                        exc,
                    )
                else:
                    logger.error(
                        "Giving up on document %s after %d attempts: %s",
                        item.document.id,
                        item.attempts,
                        exc,
                    )

        self._queue = remaining

        return succeeded
