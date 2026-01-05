from openai import OpenAI  # Official OpenAI Python SDK client (used to call responses, embeddings, etc.)
from app.core.config import settings  # Centralized config (contains your OpenAI API key)


def get_openai_client() -> OpenAI:
    # Factory function that constructs and returns an OpenAI client instance.
    #
    # What this is used for in your RAG app:
    # - Your ChatService calls:
    #     client.responses.create(...)
    #   to generate the final answer from the LLM.
    #
    # Why it’s important:
    # - Centralized configuration:
    #   You create the OpenAI client in exactly one place, so the API key (and later: base_url,
    #   organization/project IDs, timeouts, retries, etc.) are consistent everywhere.
    #
    # - Dependency injection friendly:
    #   Routes/services can call get_openai_client() (or have FastAPI inject it), which makes
    #   the code easier to test and scale. In tests, you can swap this out with a mock client.
    #
    # - Easier upgrades:
    #   If you later want to:
    #     - set a custom base_url (Azure/OpenAI compatible gateways)
    #     - add request timeouts
    #     - add tracing/logging hooks
    #     - switch keys per environment
    #   you only change this file, not your whole codebase.
    #
    # Note:
    # - This client is for *LLM generation* (your ChatService).
    # - Your embeddings are handled separately via LangChain's OpenAIEmbeddings (embeddings.py).
    return OpenAI(api_key=settings.openai_api_key)
