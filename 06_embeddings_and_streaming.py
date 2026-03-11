"""
06_embeddings.py — Gemini Embeddings for semantic search, clustering, similarity
"""
import numpy as np
import google.generativeai as genai
from config import GEMINI_EMBED, GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed(texts: list[str], task_type: str = "SEMANTIC_SIMILARITY") -> list[list[float]]:
    result = genai.embed_content(model=GEMINI_EMBED, content=texts, task_type=task_type)
    return result["embedding"]


def semantic_search(query: str, corpus: list[str], top_k: int = 3) -> list[dict]:
    """Find most similar documents to a query."""
    q_embed   = embed([query], task_type="RETRIEVAL_QUERY")[0]
    c_embeds  = embed(corpus,  task_type="RETRIEVAL_DOCUMENT")
    scored    = [
        {"text": doc, "score": cosine_similarity(q_embed, emb)}
        for doc, emb in zip(corpus, c_embeds)
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    corpus = [
        "LangGraph builds stateful multi-agent AI workflows.",
        "ChromaDB is a vector database for AI applications.",
        "Python is a general-purpose programming language.",
        "RAG combines retrieval with LLM generation for grounded answers.",
        "AWS Lambda runs serverless functions at scale.",
    ]
    results = semantic_search("How do I build agentic AI systems?", corpus)
    print("=== Semantic Search Results ===")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text']}")


# ────────────────────────────────────────────────────────────────────────────

"""
08_streaming.py — Streaming Gemini responses for real-time UX
"""

def stream_generate(prompt: str) -> None:
    """Stream tokens as they are generated."""
    import google.generativeai as genai
    from config import get_model
    model    = get_model()
    response = model.generate_content(prompt, stream=True)
    print("Streaming: ", end="", flush=True)
    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()   # newline at end


async def async_stream(prompt: str) -> None:
    """Async streaming for FastAPI / async services."""
    import asyncio
    import google.generativeai as genai
    from config import get_model
    model    = get_model()
    response = await model.generate_content_async(prompt, stream=True)
    async for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    stream_generate("Explain the attention mechanism in transformers in detail.")
