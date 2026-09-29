# Agentic AI RAG Chatbot

A Python Retrieval-Augmented Generation project based on the supplied assignment constraints. It is designed to answer questions strictly from the Agentic AI eBook and expose the answer, retrieved context, and confidence score.

## What is included

- PDF ingestion with `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- Optional OpenAI embeddings (`text-embedding-3-small`) and Pinecone vector storage.
- LangGraph workflow with `retrieve -> generate` nodes.
- FastAPI `POST /chat` endpoint returning `final_answer`, `retrieved_context`, and `confidence_score`.
- Streamlit preview with visible evidence and refusal behavior.
- Offline preview mode that works without API keys using a small built-in excerpt set.
- Six benchmark queries, including an out-of-scope FIFA question that must be refused.

## Project structure

```text
rag-agentic-ai/
├── data/
│   ├── README.md
│   └── Ebook-Agentic-AI.pdf       # add locally; not committed
├── src/
│   ├── config.py
│   ├── demo_context.py
│   ├── graph.py
│   └── ingestion.py
├── app.py                          # FastAPI app
├── streamlit_app.py                # preview UI
├── tests_sample_queries.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

Python 3.10+ is required.

```bash
python -m venv venv
source venv/bin/activate                 # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

The default `RAG_BACKEND=offline` needs no credentials and is intended for the preview. For the real assignment flow, download the supplied Agentic AI eBook as `data/Ebook-Agentic-AI.pdf`, set the API keys, and set `RAG_BACKEND=pinecone`.

## Preview

```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit. Try:

- `What is Agentic AI according to the eBook?`
- `How do AI agents differ from traditional automation systems?`
- `What role does memory play in Agentic AI workflows?`
- `Who won the 2022 FIFA World Cup?` — the assistant must refuse.

## FastAPI API

```bash
uvicorn app:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat \\
  -H 'Content-Type: application/json' \\
  -d '{"query":"What role does memory play in Agentic AI workflows?"}'
```

Example response shape:

```json
{
  "final_answer": "...",
  "retrieved_context": [{"id": "demo-4", "text": "...", "score": 0.2, "source": "demo_context"}],
  "confidence_score": 0.2
}
```

## Ingestion and Pinecone

After adding the PDF, use the ingestion helper from Python:

```python
from src.ingestion import build_pinecone_store
build_pinecone_store("data/Ebook-Agentic-AI.pdf")
```

The Pinecone index should use cosine distance and dimension 1536 for `text-embedding-3-small`. `src.graph.build_rag_graph()` uses the configured Pinecone index when both credentials are present and `RAG_BACKEND=pinecone`.

## Verification

```bash
python tests_sample_queries.py
```

The system is intentionally strict: retrieved context is exposed to the caller, generated answers must be grounded, and unsupported questions return `I cannot answer based on the provided document.`

## Architecture

```text
Question
  ↓
LangGraph StateGraph
  ↓
Retrieve top-k chunks from offline corpus or Pinecone
  ↓
Generate using only retrieved context
  ↓
Answer + context chunks + confidence score
```

## Security and limitations

- Never commit `.env`, API keys, or the source PDF unless you have permission.
- Offline confidence is a retrieval-overlap heuristic, not a calibrated probability.
- Pinecone/OpenAI mode requires valid credentials and an index populated by the ingestion step.
- The starter does not claim that the demo excerpt represents the full eBook; add the PDF and run ingestion before reporting production-quality results.
