import os
from typing import Callable, List, Tuple, Sequence

import chromadb
from chromadb.config import Settings
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import json
import tiktoken


load_dotenv()


def get_env(name: str, default: str | None = None) -> str:
    """Get an environment variable with safeguards.

    Handles the case where a UTF-8 BOM (\ufeff) accidentally prefixes the key
    name when the .env file was saved with BOM (common on Windows/Notepad).
    """
    value = os.getenv(name)
    if value is None:
        # Try BOM-prefixed variant (e.g., "\ufeffOPENROUTER_API_KEY")
        value = os.getenv("\ufeff" + name)
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_models() -> Tuple[str, str]:
    # Only need chat model now, embeddings handled by ChromaDB default
    embedding_model = "chromadb-default"  # Placeholder, not used
    chat_model = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
    return embedding_model, chat_model


def get_chunking() -> Tuple[int, int, int]:
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k = int(os.getenv("TOP_K", "3"))
    return chunk_size, overlap, top_k


def get_chroma_client() -> chromadb.Client:
    persist_dir = os.path.join(os.getcwd(), ".chromadb")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    return client


def get_session_chroma_client() -> chromadb.Client:
    """Return a session-scoped in-memory Chroma client for ephemeral uploads.

    Lives in Streamlit session_state and is discarded when the browser session ends.
    """
    key = "session_chroma_client"
    if key not in st.session_state:
        st.session_state[key] = chromadb.Client(settings=Settings(anonymized_telemetry=False))
    return st.session_state[key]


def get_openrouter_client() -> OpenAI:
    api_key = get_env("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # Optional but recommended headers for OpenRouter
    referer = os.getenv("REPO_URL", "http://localhost")
    title = os.getenv("APP_TITLE", "RAG Chatbot")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": referer,
            "X-Title": title,
        },
    )


def get_or_create_collection(name: str = "rag_docs"):
    client = get_chroma_client()
    try:
        # Use ChromaDB's default embedding function (completely free)
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(
            name=name,
            # No embedding_function specified = uses ChromaDB's built-in sentence transformers
            metadata={"hnsw:space": "cosine"},
        )


def get_or_create_session_collection(name: str):
    """Create or retrieve a collection in the session-scoped in-memory client."""
    client = get_session_chroma_client()
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(
            name=name,
            # No embedding_function specified = uses ChromaDB's built-in sentence transformers
            metadata={"hnsw:space": "cosine"},
        )


def delete_session_collection(name: str) -> None:
    """Delete a session collection if it exists (no-op on failure)."""
    try:
        client = get_session_chroma_client()
        client.delete_collection(name)
    except Exception:
        pass


def _get_encoder() -> tiktoken.Encoding:
    # Use a widely available encoding; for OpenAI-compatible models, cl100k_base fits
    return tiktoken.get_encoding("cl100k_base")


def simple_text_splitter(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Token-aware splitter using tiktoken.

    chunk_size and overlap are interpreted as token counts.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        token_slice = tokens[start:end]
        chunks.append(enc.decode(token_slice))
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks