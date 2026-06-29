"""Environment-based settings for AgentAPI."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Minimal environment config used by providers."""

    openai_api_key: str | None
    gemini_api_key: str | None
    openrouter_api_key: str | None
    default_provider: str
    postgres_host: str | None
    postgres_port: int
    postgres_user: str | None
    postgres_password: str | None
    postgres_db: str | None


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        default_provider=os.getenv("DEFAULT_PROVIDER", "openai"),
        postgres_host=os.getenv("POSTGRES_HOST"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_user=os.getenv("POSTGRES_USER"),
        postgres_password=os.getenv("POSTGRES_PASSWORD"),
        postgres_db=os.getenv("POSTGRES_DB"),
    )

