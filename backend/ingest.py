"""
ingest.py — Load and chunk the GW knowledge base corpus.
"""

import json
from pathlib import Path

CORPUS_DIR  = Path(__file__).parent.parent / "data" / "gw_corpus"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "chunks.json"

CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 80    # overlapping words between consecutive chunks


def load_documents(corpus_dir: Path) -> list:
    documents = []
    for txt_file in sorted(corpus_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        documents.append({"source": txt_file.name, "content": content})
        print(f"  Loaded: {txt_file.name} ({len(content.split())} words)")
    return documents


def chunk_document(doc: dict, chunk_size: int, overlap: int) -> list:
    words = doc["content"].split()
    chunks, start, chunk_id = [], 0, 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append({
            "id":     f"{doc['source']}__chunk_{chunk_id}",
            "source": doc["source"],
            "text":   " ".join(words[start:end]),
        })
        chunk_id += 1
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def ingest_corpus():
    print("=" * 50)
    print("GW Noise Filter — Corpus Ingestion")
    print("=" * 50)

    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")

    print(f"\nLoading documents from: {CORPUS_DIR}")
    documents = load_documents(CORPUS_DIR)
    print(f"\nTotal documents: {len(documents)}")

    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")
    return all_chunks


if __name__ == "__main__":
    ingest_corpus()
