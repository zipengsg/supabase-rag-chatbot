from langchain_openai import OpenAIEmbeddings  # LangChain wrapper that calls OpenAI to create embeddings (vectors)
from app.core.config import settings  # Centralized config (where your OpenAI API key is stored)


def get_embeddings() -> OpenAIEmbeddings:
    # Factory function that constructs and returns an OpenAIEmbeddings instance.
    #
    # What embeddings are used for in your RAG app:
    # - In ingestion:
    #   chunk text -> embedding vector -> stored in Supabase (pgvector)
    # - In retrieval:
    #   user query -> embedding vector -> compared against stored chunk vectors in Supabase
    #
    # Why it’s important:
    # - It standardizes how embeddings are created across your app (single place to configure).
    # - It makes dependency injection easy (your services/routes can call get_embeddings()).
    # - It reduces mistakes where different parts of the app accidentally use different embedding configs.
    #
    # CRITICAL NOTE (dimension matching with your Supabase function):
    # Your Supabase SQL function `match_chunks` is defined as:
    #   query_embedding vector(1536)
    # This means the embedding model you use MUST output vectors of length 1536.
    #
    # If you use a 3072-dim embedding model, ingestion/retrieval will fail with dimension mismatch.
    #
    # Recommended: explicitly set the model here so you never accidentally switch dimensions.
    # Example: OpenAIEmbeddings(model="text-embedding-3-small", ...)
    #
    # Right now, you're only passing the API key, so LangChain will use its default model
    # (which could change depending on library version / defaults).
    return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
