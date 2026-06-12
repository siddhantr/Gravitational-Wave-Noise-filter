"""
retriever.py — Semantic search over the GW FAISS vector index.
"""

import pickle
import numpy as np
from pathlib import Path
from functools import lru_cache

INDEX_FILE = Path(__file__).parent.parent / "data" / "faiss_index.pkl"


@lru_cache(maxsize=1)
def load_index() -> dict:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}\n"
            "Please run:\n"
            "  python backend/ingest.py\n"
            "  python backend/embed.py"
        )
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def retrieve(query: str, top_k: int = 5) -> list:
    """
    Retrieve top_k most relevant GW knowledge base chunks for a query.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    payload = load_index()
    model = SentenceTransformer(payload["model_name"])

    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
    query_vec = query_vec / (norm + 1e-10)

    scores, indices = payload["index"].search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = payload["chunks"][idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


if __name__ == "__main__":
    query = "What is a gravitational wave chirp signal?"
    print(f"Query: {query}\n")
    results = retrieve(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score={r['score']:.3f}) ---")
        print(f"Source: {r['source']}")
        print(f"{r['text'][:300]}...\n")
