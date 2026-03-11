# GenAI Codebase — Gemini API

A production-ready Python codebase covering core GenAI patterns using Google Gemini API.

## Structure

```
gemini_genai_codebase/
├── README.md
├── requirements.txt
├── config.py                  # Central config & env setup
├── 01_basic_chat.py           # Text generation & multi-turn chat
├── 02_rag_pipeline.py         # RAG with ChromaDB + Gemini
├── 03_structured_output.py    # JSON/Pydantic structured outputs
├── 04_multimodal.py           # Vision: image + text inputs
├── 05_function_calling.py     # Tool/function calling
├── 06_embeddings.py           # Embeddings + semantic search
├── 07_langgraph_agent.py      # LangGraph agent with Gemini
└── 08_streaming.py            # Streaming responses
```

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key-here"
```

Get your API key: https://aistudio.google.com/app/apikey
