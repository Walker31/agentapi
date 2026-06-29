from __future__ import annotations

from typing import Any
from agentapi.agent.memory import MemoryBackend

from .conversation.base import BaseConversationStore
from .vector.base import BaseVectorStore
from .types import MemoryMessage
from .embeddings.base import BaseEmbeddingProvider

class HybridMemory(MemoryBackend):
    def __init__(
        self,
        *,
        conversation_id:str,
        embedding_provider: BaseEmbeddingProvider,
        conversation_store: BaseConversationStore,
        vector_store: BaseVectorStore,
        recent_window: int = 20,
        retrieval_top_k: int = 5
    ) -> None:
        self._embedding_provider = embedding_provider
        self._conversation_store = conversation_store
        self._vector_store = vector_store
        self.conversation_id = conversation_id
        self._recent_window = recent_window
        self._retrieval_top_k = retrieval_top_k
    
    async def build_messages(self,current_query:str) -> list[dict[str,Any]]:
        recent_messages = await self._conversation_store.get_recent_messages(self.conversation_id,self.recent_window)
        query_embedding  = await self._embedding_provider.embed(current_query)
        retrieved_memories = await self._vector_store.search(self.conversation_id,query_embedding,self._retrieval_top_k)

        # TODO:
        # Merge recent conversation
        # with semantic memories.

        return []
    
    async def add(self,message:dict[str,Any]) -> None:
        memory_message = MemoryMessage.from_dict(message)
        await self._conversation_store.save_messages(self.conversation_id,memory_message)
        
        embedding = await self._embedding_provider.embed(memory_message.content)
        await self._vector_store.upsert(conversation_id = self.conversation_id,message_id= memory_message.id,embeddings=embedding,
        metadata={
            'role':memory_message.role,
            "content":memory_message.content
        })

    async def reset(self) -> None:
        await self._conversation_store.clear_conversation(self.conversation_id)
        await self._vector_store.delete(self.conversation_id)