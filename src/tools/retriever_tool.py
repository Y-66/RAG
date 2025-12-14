from functools import lru_cache
from typing import Tuple

from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain.tools import tool

from rag_workfolw import build_pipeline


@lru_cache(maxsize=1)
def _get_retriever_and_store() -> Tuple[BaseRetriever, Chroma]:
    """Lazy-build the RAG pipeline once and reuse for tool invocations."""

    retriever, vector_store, _embedding_model, _splits = build_pipeline()
    return retriever, vector_store


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query using the configured retriever."""

    retriever, _vector_store = _get_retriever_and_store()
    retrieved_docs = retriever.invoke(query)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs