"""
chain.py — RAG chain for GW physics Q&A.
Retrieves relevant GW knowledge then generates answers via LLM.
Supports Ollama (local/free) and OpenAI.
"""

import os
from backend.retriever import retrieve

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
TOP_K        = int(os.getenv("TOP_K", "5"))

SYSTEM_PROMPT = """You are GravBot, an expert AI assistant specializing in gravitational wave (GW) physics and detector science.
You explain concepts clearly, accurately, and with appropriate depth — from basic GW theory to advanced signal processing and ML denoising.

Guidelines:
- Always ground your answers in the provided knowledge base context.
- Use proper physics notation when discussing equations (e.g., h = ΔL/L, F = ma).
- Mention relevant real-world examples (GW150914, LIGO, LISA, etc.) when applicable.
- If explaining signal processing, describe the mathematical steps.
- If the context doesn't fully answer the question, say so clearly.
- Keep answers educational, precise, and structured.
- Cite which topic area your answer draws from (e.g., "From GW signal processing:...").
"""


def format_context(chunks: list) -> str:
    if not chunks:
        return "No relevant context found in the GW knowledge base."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["source"].replace("_", " ").replace(".txt", "").title()
        parts.append(f"[Context {i} — {source}]\n{chunk['text']}")
    return "\n\n".join(parts)


def call_ollama(messages: list) -> str:
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Ollama is not running. Start it with: `ollama serve`\n"
            "Then pull a model: `ollama pull llama3`\n\n"
            "Meanwhile, here's what the knowledge base says:\n\n"
            + messages[-1]["content"].split("Question:")[-1].strip()
        )
    except Exception as e:
        return f"⚠️ Ollama error: {e}"


def call_openai(messages: list) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ OPENAI_API_KEY not set in environment."
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except ImportError:
        return "⚠️ Run: pip install openai"
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"


def answer(question: str) -> dict:
    """
    Full RAG pipeline: retrieve → prompt → generate → return.
    """
    # 1. Retrieve relevant GW knowledge
    chunks = retrieve(question, top_k=TOP_K)
    context = format_context(chunks)

    # 2. Build messages
    user_message = (
        f"Use the following gravitational wave physics knowledge to answer the question.\n\n"
        f"=== Knowledge Base Context ===\n{context}\n\n"
        f"=== Question ===\n{question}"
    )

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]

    # 3. Call LLM
    if LLM_PROVIDER == "openai":
        answer_text = call_openai(messages)
    else:
        answer_text = call_ollama(messages)

    # 4. Format sources
    sources = []
    for c in chunks:
        src = c["source"].replace("_", " ").replace(".txt", "").title()
        if src not in sources:
            sources.append(src)

    return {
        "answer":  answer_text,
        "sources": sources,
        "chunks_used": len(chunks),
    }


if __name__ == "__main__":
    result = answer("What is a gravitational wave chirp signal?")
    print(result["answer"])
    print(f"\nSources: {result['sources']}")
