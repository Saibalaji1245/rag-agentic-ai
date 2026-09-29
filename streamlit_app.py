import streamlit as st
from src.graph import ask, build_rag_graph

st.set_page_config(page_title="Agentic AI RAG", page_icon="✦", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f7f8f5; }
[data-testid="stSidebar"] { background: #163943; }
.hero { padding: 1.5rem 0 1rem; }
.eyebrow { color: #1e8f86; font: 500 0.72rem monospace; letter-spacing: .12em; }
.hero h1 { font-size: 3.2rem; letter-spacing: -.06em; color: #172024; margin: .4rem 0; }
.hero p { color: #778187; font-size: 1rem; }
.answer { background: white; border: 1px solid #e2e7e5; border-left: 4px solid #1e8f86; border-radius: 10px; padding: 1.2rem; }
.context { background: white; border: 1px solid #e2e7e5; border-radius: 8px; padding: 1rem; margin: .6rem 0; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("# ✦ Agentic RAG")
st.sidebar.caption("Strictly grounded eBook assistant")
st.sidebar.divider()
st.sidebar.markdown("**Pipeline**")
st.sidebar.write("1. Retrieve relevant chunks")
st.sidebar.write("2. Generate from context only")
st.sidebar.write("3. Refuse unsupported claims")
st.sidebar.divider()
st.sidebar.caption("Offline preview mode · Pinecone/OpenAI ready")

st.markdown('<div class="hero"><div class="eyebrow">AGENTIC AI EBOOK / RAG PREVIEW</div><h1>Ask the document.<br><span style="color:#1e8f86">Trust the evidence.</span></h1><p>A strict retrieval-augmented assistant that exposes its context and confidence instead of hiding the chain of evidence.</p></div>', unsafe_allow_html=True)

query = st.text_area("Your question", placeholder="e.g. What role does memory play in Agentic AI workflows?", height=110)
examples = ["What is Agentic AI according to the eBook?", "How do AI agents differ from traditional automation systems?", "Who won the 2022 FIFA World Cup?"]
cols = st.columns(3)
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        query = example

if st.button("Retrieve and answer", type="primary", use_container_width=True) and query.strip():
    result = ask(build_rag_graph(), query.strip())
    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Final answer")
        st.markdown(f'<div class="answer">{result["answer"]}</div>', unsafe_allow_html=True)
    with right:
        st.metric("Confidence score", f'{result["score"]:.0%}')
        st.caption("Heuristic score based on retrieved context overlap in offline mode.")
    st.markdown("### Retrieved context")
    for index, item in enumerate(result["context"], 1):
        st.markdown(f'<div class="context"><b>Chunk {index} · {item["score"]:.0%} relevance</b><br>{item["text"]}<br><small>Source: {item["source"]}</small></div>', unsafe_allow_html=True)
else:
    st.info("Enter a question or choose a benchmark query to start.")
