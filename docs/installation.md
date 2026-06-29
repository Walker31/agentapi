# Installation

## Requirements

- Python 3.10+
- A provider API key (OpenAI, Gemini, or OpenRouter)

## Install from PyPI

```bash
pip install agentapi-core
```

## Install for Local Development

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install -e .
```

## Verify Installation

```bash
python -c "import agentapi; print(agentapi.__all__)"
```

## Environment Variables

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=
DEFAULT_PROVIDER=openai
POSTGRES_HOST=
POSTGRES_PORT=5432
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
```

## Notes

- `DEFAULT_PROVIDER` is used when you do not explicitly pass `provider=` to `Agent`.
- The `POSTGRES_*` variables are only required when using `PostgresConversationStore` for memory. See [Memory](memory.md#postgresconversationstore) for details.
- API key errors are surfaced as `AgentConfigurationError` with clear guidance.

