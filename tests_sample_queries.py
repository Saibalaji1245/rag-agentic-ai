"""Run the six assignment benchmark queries without external credentials."""
from src.graph import ask, build_rag_graph
from src.config import REFUSAL

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What does grounded RAG require?",
    "Who won the 2022 FIFA World Cup?",
]

if __name__ == "__main__":
    graph = build_rag_graph()
    for query in QUERIES:
        result = ask(graph, query)
        print(f"\nQ: {query}\nA: {result['answer']}\nconfidence={result['score']:.3f}")
        assert 0.0 <= result["score"] <= 1.0, "Confidence must be in [0, 1]"
    memory_result = ask(graph, QUERIES[3])
    assert "memory lets" in memory_result["answer"].lower(), "Memory query should retrieve the memory chunk"
    refusal = ask(graph, QUERIES[-1])
    assert refusal["answer"] == REFUSAL, "Out-of-scope question must be refused"
    print("\nAll benchmark checks passed.")
