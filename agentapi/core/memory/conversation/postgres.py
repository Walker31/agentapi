from __future__ import annotations
from psycopg import AsyncConnection
from .base import BaseConversationStore

class PostgresConversationStore(BaseConversationStore):
    
    def __init__(
        self,
        *,
        dsn: str,
        conversation_id: str,
    ) -> None:

        self._dsn = dsn
        self._conversation_id = conversation_id
        self._connection: AsyncConnection | None = None
        self._initialized = False

    async def _get_connection(self) -> AsyncConnection:
        """
        Lazily create and reuse a PostgreSQL connection.
        """

        if self._connection is None:
            self._connection = await AsyncConnection.connect(
                self._dsn
            )

        return self._connection

    async def _initialize_schema(self) -> None:
        conn = await self._get_connection()

        async with conn.cursor() as cursor:

            await cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (

                    id UUID PRIMARY KEY,

                    created_at TIMESTAMPTZ NOT NULL,

                    updated_at TIMESTAMPTZ NOT NULL

                );
                """
            )

            await cursor.execute(
                """
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
            )

            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_messages_conversation

                ON messages(conversation_id);
                """
            )

            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_messages_created

                ON messages(created_at DESC);
                """
            )

        await conn.commit()

    async def initialize(self) -> None:

        if self._initialized:
            return
        await self._initialize_schema()
        self._initialized = True
            