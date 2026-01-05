import os  # Used for filesystem operations: creating folders, building safe paths, writing files
from typing import List  # Type hints for lists of documents/chunks

from langchain_community.document_loaders import PyPDFLoader  # Loads PDF files into LangChain Document objects
from langchain_text_splitters import CharacterTextSplitter  # Splits text into chunks based on character counts
from langchain_core.documents import Document  # Standard LangChain container for text + metadata

from langchain_community.vectorstores import SupabaseVectorStore  # LangChain vector store wrapper backed by Supabase/Postgres
from supabase.client import Client  # Supabase Python client type
from langchain_openai import OpenAIEmbeddings  # LangChain embeddings wrapper that calls OpenAI to embed text

from app.core.config import settings  # App config (Supabase table/function names, tmp dir, etc.)


class IngestService:
    def __init__(self, supabase: Client, embeddings: OpenAIEmbeddings):
        # Create a Supabase-backed vector store using LangChain's SupabaseVectorStore.
        #
        # Under the hood, this expects you already have:
        # - a Postgres table in Supabase to store documents + embeddings (often called "documents")
        # - a SQL function (RPC) in Supabase that performs vector similarity search (often called "match_documents")
        #
        # These are configured via:
        # - settings.supabase_table: the table name to insert embeddings into
        # - settings.supabase_match_fn: the SQL function name used for similarity search
        #
        # The vector store will use `embeddings` to compute vectors for each chunk you add.
        self.vector_store = SupabaseVectorStore(
            client=supabase,                  # Supabase client instance (already authenticated)
            embedding=embeddings,             # Embedding model wrapper used to embed chunk text
            table_name=settings.supabase_table,     # Supabase table that stores vectors + text + metadata
            query_name=settings.supabase_match_fn,  # Supabase RPC/function used for similarity search
        )

    def _split(self, docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        # Create a text splitter that divides documents into chunks.
        #
        # chunk_size:
        # - Maximum size (in characters) of each chunk produced.
        #
        # chunk_overlap:
        # - Number of characters repeated between adjacent chunks.
        # - Overlap helps preserve context at chunk boundaries (prevents answers from missing split sentences).
        #
        # NOTE: CharacterTextSplitter splits purely by character count (not tokens).
        # For better semantic chunking, you might later switch to:
        # - RecursiveCharacterTextSplitter
        # - TokenTextSplitter
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Split the list of Document objects into a larger list of chunked Document objects.
        # Each output Document will usually retain metadata from the original (like page numbers).
        return splitter.split_documents(docs)

    def ingest_pdf_path(self, pdf_path: str, chunk_size: int, chunk_overlap: int) -> int:
        # Load a PDF from a filesystem path and convert it into LangChain Documents.
        #
        # PyPDFLoader typically returns one Document per page with metadata like:
        # - "source" (file path)
        # - "page" (page number)
        loader = PyPDFLoader(pdf_path)

        # Load the PDF -> list[Document] (often page-level docs).
        raw_docs = loader.load()

        # Split the raw Documents into smaller chunks for embedding + retrieval.
        chunks = self._split(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Add chunks to the vector store:
        # - computes embeddings for each chunk via OpenAIEmbeddings
        # - inserts rows into the Supabase table (text + embedding + metadata)
        #
        # IMPORTANT: this generally only stores the *chunk text* and metadata in Supabase,
        # not the original PDF file itself.
        self.vector_store.add_documents(chunks)

        # Return the number of chunks stored (useful for confirming ingestion).
        return len(chunks)

    def save_upload_to_tmp(self, filename: str, content: bytes) -> str:
        # Ensure the tmp directory exists (create it if missing).
        # settings.tmp_dir is typically something like "tmp" or "/tmp/myapp".
        os.makedirs(settings.tmp_dir, exist_ok=True)

        # Sanitize the filename to prevent directory traversal attacks.
        # Example:
        # - If user uploads "../../etc/passwd", basename() reduces it to "passwd"
        safe_name = os.path.basename(filename)

        # Build the full path to save the uploaded file into your tmp directory.
        tmp_path = os.path.join(settings.tmp_dir, safe_name)

        # Write the uploaded bytes to disk.
        # This is typically called by your API route after receiving an UploadFile.
        with open(tmp_path, "wb") as f:
            f.write(content)

        # Return the saved file path so other functions can load it (e.g., ingest_pdf_path).
        return tmp_path
