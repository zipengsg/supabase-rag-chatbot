from typing import List  # Used for type hints like List[Document]

from openai import OpenAI  # OpenAI client class (you'll pass an initialized client into ChatService)
from langchain_core.documents import Document  # LangChain document object (text + metadata per chunk)

from app.core.config import settings  # Imported settings, but currently UNUSED in this file


class ChatService:
    def __init__(self, openai_client: OpenAI):
        # Dependency injection:
        # The OpenAI client (with API key, base URL, etc.) should be created elsewhere
        # and passed in, so this service is easy to test/mock and reuse.
        self.client = openai_client

    def _build_messages(self, user_query: str, docs: List[Document]) -> List[dict]:
        # Build the "retrieved context" string by concatenating the text from each Document.
        # - d.page_content is the chunk's actual text
        # - filter `if d.page_content` avoids None/empty strings
        # - "\n\n---\n\n" visually separates chunks so the model can distinguish boundaries
        context = "\n\n---\n\n".join(d.page_content for d in docs if d.page_content)

        # System message:
        # This sets behavior/rules for the assistant.
        # In RAG, the system message typically instructs the model to rely on the retrieved context.
        system_content = (
            "You are an AI assistant with unparalleled expertise in the Agentic BaaS framework. "
            "Use the provided context to answer. If the context is insufficient, say so."
        )

        # User message:
        # We place BOTH the user's question and the retrieved context into the user content.
        # This is a simple "stuffing" pattern for RAG (works well for demos / small corpora).
        user_content = (
            f"User question:\n{user_query}\n\n"
            f"Context (retrieved):\n{context}"
        )

        # Return a list of role/content message dicts.
        # This is the format accepted by OpenAI's Responses API when using `input=messages`.
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def answer(
        self,
        query: str,
        retrieved_docs: List[Document],
        model: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        # Convert the user query + retrieved documents into the messages the LLM will see.
        messages = self._build_messages(query, retrieved_docs)

        # Call OpenAI using the "Responses API" (newer API style).
        #
        # Parameters:
        # - model: which model to use (e.g., "gpt-4.1-mini", etc.)
        # - input: the conversation/messages content
        # - max_output_tokens: caps how long the assistant's response can be
        # - temperature: controls randomness (0 = deterministic, higher = more creative)
        resp = self.client.responses.create(
            model=model,
            input=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        # `resp.output_text` is a convenience property from the SDK that returns
        # the final text output from the model (combined across output segments).
        return resp.output_text
