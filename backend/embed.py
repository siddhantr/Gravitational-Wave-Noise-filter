"""
embed.py — Generate sentence embeddings and build a FAISS vector index.
"""

import json
import pickle
import numpy as np
from pathlib import Path

CHUNKS_FILE = Path(__file__).parent.parent / "data" / "chunks.json"
INDEX_FILE  = Path(__file__).parent.parent / "data" / "faiss_index.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks() -> list:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}\n"
            "Please run `python backend/ingest.py` first."
        )
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: list):
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        raise ImportError(
            "Missing dependencies. Run:\n"
            "  pip install sentence-transformers faiss-cpu"
        )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalise for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    payload = {
        "index": index,
        "chunks": chunks,
        "embeddings": embeddings,
        "model_name": EMBEDDING_MODEL,
    }

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"Index built with {index.ntotal} vectors.")
    print(f"Saved to: {INDEX_FILE}")
    return payload


if __name__ == "__main__":
    chunks = load_chunks()
    build_index(chunks)
