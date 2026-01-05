from typing import Any, Dict, List, Optional  # Type hints for flexible payload + optional filters/thresholds

from langchain_core.documents import Document  # LangChain object: (page_content, metadata)
from supabase.client import Client  # Supabase client used to call Postgres RPC functions
from langchain_openai import OpenAIEmbeddings  # Embeddings wrapper used to embed the query

from app.core.config import settings  # Holds settings.supabase_match_fn (should be "match_chunks" for your SQL)


class RetrievalService:
    def __init__(self, supabase: Client, embeddings: OpenAIEmbeddings):
        # Store the Supabase client so we can call:
        #   supabase.rpc("<function_name>", payload).execute()
        self.supabase = supabase

        # Store embeddings so we can convert:
        #   query string -> list[float] embedding vector
        #
        # IMPORTANT (dimension gotcha):
        # Your Supabase SQL function expects query_embedding vector(1536),
        # so your embedding model MUST output 1536 dimensions (e.g., text-embedding-3-small).
        # If you use a 3072-dim model (e.g., text-embedding-3-large), you'll get dimension mismatch errors.
        self.embeddings = embeddings

    def similarity_search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        match_threshold: Optional[float] = None,
    ) -> List[Document]:
        # Embed the user query into a vector (list[float]).
        # This is what Supabase/pgvector will compare against stored chunk embeddings.
        query_embedding = self.embeddings.embed_query(query)

        # Build the RPC payload to match your Postgres function signature exactly:
        #
        # SQL signature:
        #   match_chunks(
        #     query_embedding vector(1536),
        #     match_threshold float default 0.0,
        #     match_count int default 5,
        #     filter jsonb default '{}'::jsonb
        #   )
        #
        # Therefore, your payload keys MUST be:
        # - "query_embedding"
        # - "match_count"
        # - "filter"
        # - optionally "match_threshold"
        payload: Dict[str, Any] = {
            "query_embedding": query_embedding,  # vector(1536) expected by SQL function
            "match_count": k,                    # top-k results
            "filter": filter or {},              # jsonb metadata filter; {} means "no filtering"
        }

        # Only include match_threshold if explicitly provided.
        # If omitted, the SQL default (0.0) applies.
        #
        # match_threshold logic in SQL:
        #   similarity = 1 - cosine_distance
        #   keep rows where similarity >= match_threshold
        if match_threshold is not None:
            payload["match_threshold"] = match_threshold

        # Call the Supabase RPC function.
        #
        # IMPORTANT:
        # settings.supabase_match_fn MUST be "match_chunks" (the function you showed),
        # not "match_documents" or anything else, otherwise you'll get "function not found".
        resp = self.supabase.rpc(settings.supabase_match_fn, payload).execute()

        # RPC returns a list of dict rows (or None).
        # Your function returns columns:
        #   id, content, metadata, similarity
        rows = resp.data or []

        # Convert returned rows into LangChain Document objects.
        #
        # - page_content: we use "content" because your SQL returns "content".
        #   (we keep a fallback to "page_content" for compatibility with other schemas)
        # - metadata: we keep the stored metadata AND enrich it with:
        #   - _id: row id (uuid) for traceability/citations
        #   - _similarity: similarity score returned by SQL (useful for debugging + ranking display)
        #
        # Note: storing _similarity in metadata is convenient because Document only has
        # (page_content, metadata) by default.
        return [
            Document(
                page_content=r.get("content") or r.get("page_content") or "",
                metadata={
                    **(r.get("metadata") or {}),
                    "_id": r.get("id"),
                    "_similarity": r.get("similarity"),
                },
            )
            for r in rows
        ]
