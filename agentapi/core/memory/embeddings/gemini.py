from __future__ import annotations

import httpx

from agentapi.errors import AgentProviderError
from agentapi.core.memory.embeddings.base import BaseEmbeddingProvider
from agentapi.core.memory.embeddings.types import Embedding


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Generates embeddings via the Gemini embedContent API.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-004",
        timeout: float = 30.0,
    ) -> None:

        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        self.base_url = (
            "https://generativelanguage.googleapis.com/v1beta"
        )

        self._client = httpx.AsyncClient(
            timeout=self.timeout,
        )

    @property
    def _endpoint(self) -> str:

        return (
            f"{self.base_url}"
            f"/models/{self.model}:embedContent"
        )

    def _build_payload(
        self,
        text: str,
    ) -> dict:

        return {
            "model": f"models/{self.model}",
            "content": {
                "parts": [
                    {
                        "text": text,
                    }
                ]
            },
        }

    async def embed(
        self,
        text: str,
    ) -> Embedding:
        """
        Generate an embedding for a single piece of text.
        """

        response = await self._client.post(
            self._endpoint,
            params={
                "key": self.api_key,
            },
            json=self._build_payload(text),
        )

        if response.status_code != 200:
            detail = response.text.strip()[:500] if response.text else "Unknown error"
            raise AgentProviderError(
                f"Gemini embedding request failed ({response.status_code}) "
                f"for model '{self.model}'. Response: {detail}",
                status_code=response.status_code,
            )

        data = response.json()

        vector = data["embedding"]["values"]

        return Embedding(
            vector=vector,
            model=self.model,
            dimensions=len(vector),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""

        await self._client.aclose()