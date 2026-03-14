"""
02_rag_pipeline.py — Production RAG pipeline: ingest → chunk → embed → retrieve → generate
Covers: document chunking, ChromaDB vector store, hybrid retrieval, RAGAS-style evaluation
"""
import re
import uuid
import chromadb
import google.generativeai as genai
from dataclasses import dataclass, field
from config import get_model, GEMINI_FLASH, GEMINI_EMBED


# ── Data Models ────────────────────────────────────────────────────────────
@dataclass
class Document:
    content:  str
    metadata: dict = field(default_factory=dict)
    doc_id:   str  = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RAGResponse:
    answer:          str
    source_chunks:   list[str]
    source_metadata: list[dict]
    confidence:      float = 0.0


# ── Stage 1: Chunking ──────────────────────────────────────────────────────
#cbunking based on overlapping so that semantics wont last at the end and beginning 
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Sliding-window character-level chunker.
    For production: use RecursiveCharacterTextSplitter (LangChain) or semantic chunking.
    """
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]   # drop tiny trailing chunks


def sentence_aware_chunk(text: str, max_sentences: int = 5) -> list[str]:
    """Chunk by sentence boundaries — better for QA tasks."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [
        " ".join(sentences[i:i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]


# ── Stage 2: Embedding via Gemini ──────────────────────────────────────────
def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    task_type options:
      RETRIEVAL_DOCUMENT  — for indexing chunks
      RETRIEVAL_QUERY     — for query-time embedding
      SEMANTIC_SIMILARITY — for similarity scoring
    """
    result = genai.embed_content(
        model=GEMINI_EMBED,
        content=texts,
        task_type=task_type,
    )
    return result["embedding"] if isinstance(texts, str) else result["embedding"]


# ── Stage 3: ChromaDB Vector Store ────────────────────────────────────────
class VectorStore:
    def __init__(self, collection_name: str = "rag_collection"):
        self.client     = chromadb.Client()   # In-memory; use PersistentClient for disk
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )

    def add_documents(self, documents: list[Document]) -> None:
        all_chunks, all_ids, all_metas, all_embeds = [], [], [], []
        for doc in documents:
            chunks = chunk_text(doc.content)
            embeds = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")
            for i, (chunk, embed) in enumerate(zip(chunks, embeds)):
                all_chunks.append(chunk)
                all_ids.append(f"{doc.doc_id}_{i}")
                all_metas.append({**doc.metadata, "doc_id": doc.doc_id, "chunk_idx": i})
                all_embeds.append(embed)

        self.collection.add(
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metas,
            embeddings=all_embeds,
        )
        print(f"✅ Indexed {len(all_chunks)} chunks from {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        query_embed = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
        results = self.collection.query(
            query_embeddings=[query_embed],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return {
            "chunks":    results["documents"][0],
            "metadata":  results["metadatas"][0],
            "distances": results["distances"][0],
        }


# ── Stage 4: Generation with Retrieved Context ─────────────────────────────
RAG_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions strictly based on the provided context.
Rules:
- Only use information from the context below.
- If the answer is not in the context, say "I don't have enough information to answer that."
- Be concise and cite which part of the context you used.
- Never hallucinate or add information not present in context.
"""

def generate_rag_answer(query: str, context_chunks: list[str]) -> str:
    model   = get_model(GEMINI_FLASH, system_prompt=RAG_SYSTEM_PROMPT)
    context = "\n\n---\n\n".join(
        [f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)]
    )
    prompt = f"""Context:
{context}

Question: {query}

Answer:"""
    return model.generate_content(prompt).text


# ── Stage 5: Full RAG Pipeline ─────────────────────────────────────────────
class RAGPipeline:
    def __init__(self, collection_name: str = "rag_collection"):
        self.store = VectorStore(collection_name)
        self.model = get_model(GEMINI_FLASH, system_prompt=RAG_SYSTEM_PROMPT)

    def ingest(self, documents: list[Document]) -> None:
        self.store.add_documents(documents)

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        # Retrieve
        results   = self.store.retrieve(question, top_k=top_k)
        chunks    = results["chunks"]
        metadata  = results["metadata"]
        distances = results["distances"]

        # Confidence: convert cosine distance → similarity score
        avg_similarity = 1 - (sum(distances) / len(distances)) if distances else 0.0

        # Generate
        answer = generate_rag_answer(question, chunks)

        return RAGResponse(
            answer=answer,
            source_chunks=chunks,
            source_metadata=metadata,
            confidence=round(avg_similarity, 3),
        )


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample documents (replace with your actual corpus)
    docs = [
        Document(
            content="""
            LangGraph is a library for building stateful, multi-actor applications with LLMs.
            It extends LangChain to support cyclic graphs, enabling complex agent workflows.
            Key features include: persistent state management, conditional edges, human-in-the-loop,
            and streaming support. LangGraph uses a directed graph where nodes are Python functions
            and edges define control flow.
            """,
            metadata={"source": "langraph_docs", "topic": "agents"}
        ),
        Document(
            content="""
            ChromaDB is an open-source vector database optimized for AI applications.
            It supports in-memory and persistent storage, cosine/L2/IP distance metrics,
            and metadata filtering. ChromaDB integrates natively with LangChain, LlamaIndex,
            and other AI frameworks. For production, use the PersistentClient or hosted Chroma Cloud.
            """,
            metadata={"source": "chromadb_docs", "topic": "vector_store"}
        ),
    ]

    pipeline = RAGPipeline()
    pipeline.ingest(docs)

    question = "How does LangGraph handle state in agent workflows?"
    result   = pipeline.query(question)

    print(f"Q: {question}")
    print(f"A: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {[m['source'] for m in result.source_metadata]}")
