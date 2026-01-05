from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# WHAT THIS FILE IS FOR (high-level)
# -------------------------------------------------------------------
# These Pydantic models are used by FastAPI as:
# 1) Request validation:
#    - If a user sends bad input (empty query, k too large, invalid threshold),
#      FastAPI automatically returns a 422 error with details.
#
# 2) Response serialization:
#    - Ensures your endpoints consistently return the expected fields/types.
#
# 3) Auto-generated API docs (Swagger UI at /docs):
#    - FastAPI uses these models to generate interactive documentation and an OpenAPI schema.


class IngestResponse(BaseModel):
    # -------------------------------------------------------------------
    # Used as the response model for your ingestion endpoint (e.g. POST /ingest).
    # -------------------------------------------------------------------
    # After you upload a PDF and chunk+embed it, your API returns:
    # - how many chunks were inserted,
    # - which Supabase table they went into,
    # - which RPC function is configured for searching those chunks.
    #
    # Why important:
    # - Confirms ingestion actually worked (chunks_added > 0).
    # - Helps debug configuration mistakes (wrong table name or RPC function).
    # - Useful when you support multiple datasets/tables later.
    chunks_added: int
    table_name: str
    match_function: str


class ChatRequest(BaseModel):
    # -------------------------------------------------------------------
    # Used as the request body model for your chat endpoint (e.g. POST /chat).
    # -------------------------------------------------------------------
    # This represents everything the backend needs to:
    # 1) Retrieve relevant chunks from Supabase
    # 2) Call the LLM with a grounded prompt
    # 3) Return an answer (+ sources)
    #
    # Why important:
    # - Makes your endpoint deterministic and testable:
    #   same query + same retrieval params -> similar retrieval -> comparable outputs.
    # - Prevents abuse / runaway costs by bounding parameters (k, tokens, temperature).
    # - Lets power users adjust retrieval strategy (filter/threshold) without changing code.

    # The user's question; cannot be empty.
    query: str = Field(..., min_length=1)

    # Number of chunks to retrieve (top-k).
    # Bounded to prevent huge prompt stuffing and excessive DB/embedding load.
    k: int = Field(4, ge=1, le=20)

    # Optional RPC parameters ---------------------------------------------------
    #
    # Maps directly to your Supabase SQL function `match_chunks(..., filter jsonb)`.
    # This is used in SQL as:
    #   where c.metadata @> filter
    #
    # Why important:
    # - Enables scoped retrieval:
    #   e.g., only search within a specific file/user/tenant/project.
    # - Critical for multi-tenant systems and for reducing irrelevant context.
    filter: Optional[Dict[str, Any]] = None

    # Maps to your Supabase SQL function's match_threshold:
    #   and similarity >= match_threshold
    #
    # Why important:
    # - Prevents low-quality context from being passed to the model.
    # - If set too high, you may retrieve nothing (so keep default None or low values).
    match_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

    # LLM parameters -----------------------------------------------------------
    #
    # These allow per-request overrides. If None, you usually fall back to app defaults.

    # Model selection per request (useful for testing, cost control, or quality upgrades).
    model: Optional[str] = None

    # Output length cap (cost + latency control).
    max_output_tokens: Optional[int] = Field(None, ge=50, le=4000)

    # Creativity control (0 = more deterministic; higher = more varied).
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class SourceChunk(BaseModel):
    # -------------------------------------------------------------------
    # Used inside ChatResponse.sources[] to represent each retrieved chunk.
    # -------------------------------------------------------------------
    # Why important:
    # - Transparency: users can see what text the model relied on.
    # - Debugging: helps you understand bad answers (wrong chunks retrieved).
    # - Auditability: essential for enterprise/legal/compliance use cases.
    #
    # Typically:
    # - content_preview is a shortened snippet of the chunk for UI display
    # - metadata includes provenance like filename/page/chunk_id
    #   and can include debug fields like "_similarity" and "_id"
    content_preview: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    # -------------------------------------------------------------------
    # Used as the response model for your chat endpoint (e.g. POST /chat).
    # -------------------------------------------------------------------
    # Why important:
    # - answer: what the user reads
    # - sources: what makes the answer trustworthy and explainable
    #
    # In RAG, returning sources is one of the biggest quality multipliers:
    # it lets you verify grounding, build UI citations, and troubleshoot retrieval.
    answer: str
    sources: List[SourceChunk]
