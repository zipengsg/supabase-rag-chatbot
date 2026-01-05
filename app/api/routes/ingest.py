import os  # Used to check if the temp file exists and to delete it during cleanup
from fastapi import APIRouter, File, UploadFile, HTTPException  # FastAPI routing + file upload primitives

from app.core.config import settings  # App config: tmp dir, chunk params, supabase table/function names
from app.models.schemas import IngestResponse  # Pydantic response schema for this endpoint
from app.clients.supabase_client import get_supabase_client  # Factory for Supabase client
from app.clients.embeddings import get_embeddings  # Factory for OpenAI embeddings wrapper
from app.services.ingest_service import IngestService  # Service that saves uploads + chunks/embeds/inserts

# Create a router group for ingestion endpoints.
# - prefix="/ingest" means all routes here start with /ingest
# - tags=["ingest"] groups these endpoints in Swagger UI
router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    # UploadFile is FastAPI's streamed file object (doesn't load whole file into memory automatically).
    # File(...) tells FastAPI this parameter comes from multipart/form-data.
    file: UploadFile = File(...),

    # Default chunking parameters come from settings, but can be overridden per request.
    # These values control how your PDF is split before embedding.
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,

    # If keep_file=False, the uploaded PDF will be deleted from tmp after ingestion finishes.
    # This matters because:
    # - Many RAG pipelines only store chunks+embeddings in the DB, not the raw PDF.
    # - Keeping files on disk can fill the server storage quickly.
    keep_file: bool = False,
):
    # -------------------------------------------------------------------------
    # Purpose of this endpoint:
    # 1) Accept a PDF upload from the client
    # 2) Save it to a temporary folder on the server (so a PDF loader can read it)
    # 3) Load PDF -> split into chunks -> embed -> insert into Supabase vector table
    # 4) Optionally delete the temp file (default behavior)
    #
    # Why it’s important:
    # - This endpoint is the "knowledge loading" step of your RAG system.
    # - Without ingestion, retrieval has nothing to search, so chat will hallucinate or be empty.
    # - Chunking and embedding choices made here directly determine retrieval quality later.
    # -------------------------------------------------------------------------

    # Create clients needed for ingestion.
    # NOTE: You are constructing these inside the request handler.
    # For bigger apps, you typically use FastAPI dependency injection (Depends),
    # but this is fine for a simple project.
    supabase = get_supabase_client()
    embeddings = get_embeddings()

    # Create a service that:
    # - writes temp files
    # - uses LangChain SupabaseVectorStore to add documents (chunks) into Supabase
    ingest_service = IngestService(supabase=supabase, embeddings=embeddings)

    # Read the uploaded file bytes.
    # UploadFile supports async reads; this loads the entire file into memory as bytes.
    # For very large PDFs, you'd want streaming, but for a demo this is typical.
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}")

    # Persist the uploaded PDF to your tmp folder so PyPDFLoader can open it by path.
    # This does NOT mean the PDF will stay there permanently—see the finally block below.
    tmp_path = ingest_service.save_upload_to_tmp(file.filename, content)

    try:
        # Ingest the PDF from disk:
        # - load PDF pages into LangChain Documents
        # - split into chunks (chunk_size / chunk_overlap)
        # - embed each chunk using OpenAI embeddings
        # - insert chunk text + metadata + embedding vector into Supabase (e.g., public.chunks)
        chunks_added = ingest_service.ingest_pdf_path(
            tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Return a structured response confirming:
        # - how many chunks were created
        # - which table they were stored in
        # - which RPC function will be used later for similarity search
        #
        # IMPORTANT:
        # settings.supabase_match_fn should match your Supabase SQL function name:
        #   create or replace function public.match_chunks(...)
        return IngestResponse(
            chunks_added=chunks_added,
            table_name=settings.supabase_table,
            match_function=settings.supabase_match_fn,
        )

    except Exception as e:
        # Any exception during parsing/splitting/embedding/db insert becomes a 500.
        # In production you might:
        # - log full stack trace
        # - return a cleaner message
        # - handle known failures separately (e.g., invalid PDF, dimension mismatch, RPC issues)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")

    finally:
        # Cleanup step:
        # By default keep_file=False, so the server deletes the temp PDF after ingestion.
        #
        # Why you might NOT see the uploaded file in tmp after success:
        # - Because this finally block removes it right away unless keep_file=True.
        # - This is often desired (avoid disk bloat, reduce risk of storing sensitive files).
        #
        # If you're debugging and want to inspect tmp files, call endpoint with keep_file=true.
        if not keep_file:  # <-- conditional cleanup
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                # Swallow cleanup errors so they don't mask the real ingestion result.
                # (e.g., file already deleted, permission issue, etc.)
                pass
