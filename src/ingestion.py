"""PDF ingestion and vector-index setup for the assignment."""
from pathlib import Path
from typing import Any

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, PINECONE_INDEX_NAME


def load_and_split_pdf(pdf_path: str | Path):
    """Load and chunk the source PDF using LangChain components."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def build_pinecone_store(pdf_path: str | Path, index_name: str = PINECONE_INDEX_NAME):
    """Embed and upsert PDF chunks into Pinecone when credentials are configured."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    chunks = load_and_split_pdf(pdf_path)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)


def chunk_summary(pdf_path: str | Path) -> dict[str, Any]:
    chunks = load_and_split_pdf(pdf_path)
    return {"pages_or_chunks": len(chunks), "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}
