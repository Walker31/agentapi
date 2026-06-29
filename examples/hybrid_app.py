"""
Example application using HybridMemory with Postgres and Qdrant.

Requires:
- PostgreSQL running (set POSTGRES_HOST, etc.)
- Qdrant running (set QDRANT_URL)
- GEMINI_API_KEY in .env

To run:
    uvicorn examples.hybrid_app:app --reload
"""

import os
from uuid import uuid4
from dotenv import load_dotenv

from agentapi import Agent, AgentAPI
from agentapi.core.memory.hybrid import HybridMemory
from agentapi.core.memory.conversation.postgres import PostgresConversationStore
from agentapi.core.memory.vector.qdrant import QdrantVectorStore
from agentapi.core.memory.embeddings.gemini import GeminiEmbeddingProvider

# Load environment variables from .env
load_dotenv()

# We generate a static conversation ID so the same memory is maintained 
# across restarts while testing the script. 
# For a real application, you would create this dynamically per-user/session.
CONVERSATION_ID = "00000000-0000-0000-0000-000000000001"

# 1. Initialize the Conversation Store (PostgreSQL)
postgres_store = PostgresConversationStore(
    conversation_id=CONVERSATION_ID,
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    dbname=os.getenv("POSTGRES_DB", "postgres"),
)

# 2. Initialize the Vector Store (Qdrant)
qdrant_store = QdrantVectorStore(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    collection_name="example_hybrid_memory_v2",
    vector_size=3072, 
)

# 3. Initialize the Embedding Provider (Gemini)
gemini_provider = GeminiEmbeddingProvider(
    model="gemini-embedding-001",
    api_key=os.environ.get("GEMINI_API_KEY", ""),
)

# 4. Wire everything together via HybridMemory
hybrid_memory = HybridMemory(
    conversation_store=postgres_store,
    vector_store=qdrant_store,
    embedding_provider=gemini_provider,
    conversation_id=CONVERSATION_ID,
    recent_window=5,      # Fetch last 5 messages from postgres
    retrieval_top_k=3,    # Fetch top 3 semantically relevant older memories
)

# 5. Provide memory to the Agent
agent = Agent(
    system_prompt=(
        "You are a helpful AI assistant with a flawless memory. "
        "You always remember the user's past statements because of your "
        "hybrid memory pipeline (Postgres + Qdrant)."
    ),
    provider="gemini",
    memory=hybrid_memory,
)

# 6. Expose via AgentAPI
app = AgentAPI()

from fastapi import Request

@app.chat("/chat")
async def chat(request: Request):
    """
    Chat endpoint.
    Example:
    curl -X POST http://localhost:8000/chat -H "Content-Type: text/plain" -d "I bought a MacBook Pro"
    """
    message = (await request.body()).decode("utf-8")
    return await agent.run(message)

@app.chat("/reset")
async def reset(message: str):
    """Clear both PostgreSQL conversation history and Qdrant vectors."""
    await hybrid_memory.reset()
    return "Memory cleared."
