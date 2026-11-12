
import sqlite3
import json
import math
import hashlib

from pathlib import Path
from datetime import datetime, timezone

from embeddings import generate_embeddings, generate_embedding


# ==========================================
# 1. DATABASE CONFIGURATION
# ==========================================

DATABASE_PATH = Path(__file__).parent / "nexum_knowledge.db"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


# ==========================================
# 2. DATABASE INITIALIZATION
# ==========================================

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                content_hash TEXT
            )
        """)

        # Safely migrate an existing database
        cursor.execute("PRAGMA table_info(knowledge)")

        existing_columns = [
            column[1] for column in cursor.fetchall()
        ]

        if "chunk_index" not in existing_columns:
            cursor.execute("""
                ALTER TABLE knowledge
                ADD COLUMN chunk_index INTEGER DEFAULT 0
            """)

        if "content_hash" not in existing_columns:
            cursor.execute("""
                ALTER TABLE knowledge
                ADD COLUMN content_hash TEXT
            """)

        # Prevent duplicate chunks for the same source
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_knowledge_source_hash
            ON knowledge(source, content_hash)
        """)

        connection.commit()

        print("NEXUM knowledge database initialized.")

    finally:
        connection.close()


# ==========================================
# 3. SPLIT DOCUMENT INTO CHUNKS
# ==========================================

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):

    text = text.strip()

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        # Prefer ending at a newline when possible
        if end < len(text):

            newline_position = text.rfind(
                "\n",
                start + chunk_size // 2,
                end
            )

            if newline_position > start:
                end = newline_position

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


# ==========================================
# 4. STORE DOCUMENT AND EMBEDDINGS
# ==========================================

async def add_document(source: str, content: str):

    source = source.strip()
    content = content.strip()

    if not source:
        raise ValueError("Source cannot be empty.")

    if not content:
        raise ValueError("Document content cannot be empty.")

    # Split document into smaller pieces
    chunks = split_into_chunks(content)

    if not chunks:
        raise ValueError("No document chunks were generated.")

    # Generate embeddings for all chunks
    vectors = await generate_embeddings(chunks)

    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        cursor = connection.cursor()

        # Replace previous version of this source.
        # This prevents duplicate records when the same file
        # is uploaded again and removes outdated chunks.
        cursor.execute(
            "DELETE FROM knowledge WHERE source = ?",
            (source,)
        )

        created_at = datetime.now(timezone.utc).isoformat()

        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors)
        ):

            chunk_hash = hashlib.sha256(
                chunk.encode("utf-8")
            ).hexdigest()

            cursor.execute("""
                INSERT INTO knowledge (
                    source,
                    content,
                    embedding,
                    created_at,
                    chunk_index,
                    content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                source,
                chunk,
                json.dumps(vector),
                created_at,
                index,
                chunk_hash
            ))

        connection.commit()

        return {
            "id": cursor.lastrowid,
            "source": source,
            "chunks_created": len(chunks),
            "message": "Document indexed successfully."
        }

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ==========================================
# 5. COSINE SIMILARITY
# ==========================================

def cosine_similarity(vector_a, vector_b):

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ==========================================
# 6. SEARCH KNOWLEDGE DATABASE
# ==========================================

async def search_documents(question: str, limit: int = 5):

    if not question.strip():
        raise ValueError("Search question cannot be empty.")

    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    # Generate question embedding
    question_vector = await generate_embedding(question)

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                id,
                source,
                content,
                embedding,
                created_at,
                chunk_index
            FROM knowledge
        """)

        documents = cursor.fetchall()

    finally:
        connection.close()

    results = []

    for document in documents:

        (
            document_id,
            source,
            content,
            embedding_json,
            created_at,
            chunk_index
        ) = document

        document_vector = json.loads(embedding_json)

        similarity = cosine_similarity(
            question_vector,
            document_vector
        )

        results.append({
            "id": document_id,
            "source": source,
            "content": content,
            "chunk_index": chunk_index,
            "similarity": round(similarity, 4),
            "created_at": created_at
        })

    # Sort by highest similarity
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:limit]


# ==========================================
# 7. DATABASE STARTUP
# ==========================================

initialize_database()
