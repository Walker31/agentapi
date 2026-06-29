from __future__ import annotations

from datetime import datetime, timezone

from psycopg import AsyncConnection
from psycopg.types.json import Json

from agentapi.config.settings import get_settings
from agentapi.core.memory.document import MemoryDocument
from .base import BaseConversationStore

# ---------------------------------------------------------------------------
# Named SQL constants
# ---------------------------------------------------------------------------

_CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
"""

_CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (conversation_id)
        REFERENCES conversations(id)
        ON DELETE CASCADE
);
"""

_CREATE_MESSAGES_INDEX = """
CREATE INDEX IF NOT EXISTS
idx_messages_conversation_created
ON messages(conversation_id, created_at DESC);
"""

_UPSERT_CONVERSATION = """
INSERT INTO conversations (id, created_at, updated_at)
VALUES (%s, %s, %s)
ON CONFLICT (id) DO NOTHING;
"""

_INSERT_MESSAGE = """
INSERT INTO messages (
    id,
    conversation_id,
    role,
    content,
    metadata,
    created_at
)
VALUES (%s, %s, %s, %s, %s, %s);
"""

_UPDATE_CONVERSATION_TIMESTAMP = """
UPDATE conversations
SET updated_at = %s
WHERE id = %s;
"""

_DELETE_CONVERSATION = """
DELETE FROM conversations
WHERE id = %s;
"""

_SELECT_RECENT_MESSAGES = """
SELECT
    id,
    role,
    content,
    metadata,
    created_at
FROM messages
WHERE conversation_id = %s
ORDER BY created_at DESC
LIMIT %s;
"""


class PostgresConversationStore(BaseConversationStore):
    """Persistent conversation store backed by PostgreSQL (psycopg 3)."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        dbname: str | None = None,
        conversation_id: str,
    ) -> None:
        settings = get_settings()

        self._host = host or settings.postgres_host
        self._port = port or settings.postgres_port
        self._user = user or settings.postgres_user
        self._password = password or settings.postgres_password
        self._dbname = dbname or settings.postgres_db

        if not self._host:
            raise ValueError(
                "No PostgreSQL host provided. "
                "Pass host= explicitly or set POSTGRES_HOST in .env"
            )

        self._conversation_id = conversation_id
        self._connection: AsyncConnection | None = None

    def _build_conninfo(self) -> str:
        """Build a libpq connection string from individual parameters."""

        parts = [f"host={self._host}"]

        if self._port:
            parts.append(f"port={self._port}")
        if self._user:
            parts.append(f"user={self._user}")
        if self._password:
            parts.append(f"password={self._password}")
        if self._dbname:
            parts.append(f"dbname={self._dbname}")

        return " ".join(parts)

    async def _get_connection(self) -> AsyncConnection:
        """Return a live connection, reconnecting if necessary."""

        if self._connection is None or self._connection.closed:
            self._connection = await AsyncConnection.connect(
                self._build_conninfo(),
                autocommit=True,
            )

        return self._connection


    async def _initialize_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""

        conn = await self._get_connection()

        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(_CREATE_CONVERSATIONS_TABLE)
                await cursor.execute(_CREATE_MESSAGES_TABLE)
                await cursor.execute(_CREATE_MESSAGES_INDEX)

    async def initialize(self) -> None:
        """Initialize the PostgreSQL schema.

        This method is idempotent — it is always safe to call
        more than once, even from concurrent coroutines.
        """

        await self._initialize_schema()

    async def create_conversation(
        self,
    ) -> None:
        """Create the conversation if it does not already exist."""

        now = datetime.now(timezone.utc)
        conn = await self._get_connection()

        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(
                    _UPSERT_CONVERSATION,
                    (self._conversation_id, now, now),
                )


    async def save_document(
        self,
        document: MemoryDocument,
    ) -> None:
        """Persist a conversation document and update the conversation timestamp."""

        await self.initialize()
        await self.create_conversation()

        now = datetime.now(timezone.utc)
        conn = await self._get_connection()

        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(
                    _INSERT_MESSAGE,
                    (
                        document.id,
                        self._conversation_id,
                        document.role,
                        document.content,
                        Json(document.metadata),
                        document.timestamp,
                    ),
                )

                await cursor.execute(
                    _UPDATE_CONVERSATION_TIMESTAMP,
                    (now, self._conversation_id),
                )

    async def get_recent_documents(
        self,
        limit: int,
    ) -> list[MemoryDocument]:
        """Retrieve the most recent documents in chronological order."""

        await self.initialize()

        conn = await self._get_connection()

        async with conn.cursor() as cursor:
            await cursor.execute(
                _SELECT_RECENT_MESSAGES,
                (self._conversation_id, limit),
            )

            rows = await cursor.fetchall()

        return [
            MemoryDocument(
                id=row[0],
                conversation_id=self._conversation_id,
                role=row[1],
                content=row[2],
                metadata=row[3],
                timestamp=row[4],
            )
            for row in reversed(rows)
        ]

    async def clear_conversation(self) -> None:
        """Delete the conversation and all its messages (via CASCADE)."""

        await self.initialize()

        conn = await self._get_connection()

        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(
                    _DELETE_CONVERSATION,
                    (self._conversation_id,),
                )


    async def close(self) -> None:
        """Close the database connection, if open."""

        if self._connection is None:
            return

        await self._connection.close()
        self._connection = None