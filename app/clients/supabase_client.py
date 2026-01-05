from supabase.client import Client, create_client  # Supabase Python client + factory function
from app.core.config import settings  # Centralized config (Supabase URL + service key / anon key)


def get_supabase_client() -> Client:
    # Factory function that constructs and returns a Supabase client instance.
    #
    # What this is used for in your RAG app:
    # - In ingestion:
    #   - LangChain's SupabaseVectorStore uses this client to INSERT rows into your vector table
    #     (e.g., public.chunks), storing content + metadata + embedding vectors.
    #
    # - In retrieval:
    #   - Your RetrievalService calls:
    #       supabase.rpc("match_chunks", payload).execute()
    #     to run your Postgres SQL function that performs vector similarity search.
    #
    # Why it’s important:
    # - Centralizes configuration:
    #   If your Supabase project URL/key changes (dev vs prod), you update it in one place.
    #
    # - Makes dependency injection easy:
    #   Services/routes can use the same client instance, or you can easily mock it in tests.
    #
    # - Keeps your code modular:
    #   You avoid sprinkling create_client(...) calls across many files.
    #
    # SECURITY NOTE (very important):
    # - settings.supabase_key is often either:
    #   1) anon key (safe-ish for client-side apps, limited by RLS), OR
    #   2) service_role key (FULL ACCESS, bypasses RLS; must stay server-side only)
    #
    # In a backend RAG API like yours, people often use the service_role key so the server
    # can insert/read regardless of user identity. That is fine ONLY if:
    # - the key is never exposed to the browser
    # - you keep it in environment variables on the server
    #
    # If you ever move Supabase calls to the frontend, DO NOT use the service_role key there.
    return create_client(settings.supabase_url, settings.supabase_key)
