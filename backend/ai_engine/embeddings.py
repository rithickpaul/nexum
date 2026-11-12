
import httpx
from fastapi import HTTPException

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


async def generate_embedding(text: str) -> list[float]:
    """
    Convert text into a numerical embedding using Nomic.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text
                }
            )

            response.raise_for_status()
            result = response.json()

            return result["embeddings"][0]

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama."
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Embedding generation timed out."
        )

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama embedding error: {error.response.text}"
        )

    except (KeyError, IndexError):
        raise HTTPException(
            status_code=502,
            detail="Invalid embedding response from Ollama."
        )
