from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import ask, build_rag_graph

app = FastAPI(title="Agentic AI RAG API", version="0.1.0")
graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[dict]
    confidence_score: float


@app.get("/health")
def health():
    return {"status": "ok", "backend": type(graph).__name__}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        result = ask(graph, request.query.strip())
        return QueryResponse(final_answer=result["answer"], retrieved_context=result["context"], confidence_score=result["score"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
