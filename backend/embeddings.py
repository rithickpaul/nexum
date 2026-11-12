
import httpx
from typing import List


# ==========================================
# 1. NEXUM EMBEDDING CONFIGURATION
# ==========================================

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


# ==========================================
# 2. GENERATE EMBEDDING FOR ONE TEXT
# ==========================================

async def generate_embedding(text: str) -> List[float]:

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:

            response = await client.post(
                OLLAMA_URL,
                json=payload
            )

            response.raise_for_status()

            result = response.json()

            embedding = result["embeddings"][0]

            return embedding

    except httpx.ConnectError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        )

    except httpx.TimeoutException:
        raise RuntimeError(
            "Embedding generation timed out."
        )

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"Ollama embedding error: {error.response.text}"
        )

    except (KeyError, IndexError, ValueError):
        raise RuntimeError(
            "Invalid embedding response from Ollama."
        )


# ==========================================
# 3. GENERATE EMBEDDINGS FOR MULTIPLE TEXTS
# ==========================================

async def generate_embeddings(texts: List[str]) -> List[List[float]]:

    if not texts:
        return []

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:

            response = await client.post(
                OLLAMA_URL,
                json=payload
            )

            response.raise_for_status()

            result = response.json()

            embeddings = result["embeddings"]

            if len(embeddings) != len(texts):
                raise RuntimeError(
                    "Embedding count does not match input count."
                )

            return embeddings

    except httpx.ConnectError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        )

    except httpx.TimeoutException:
        raise RuntimeError(
            "Batch embedding generation timed out."
        )

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"Ollama embedding error: {error.response.text}"
        )
