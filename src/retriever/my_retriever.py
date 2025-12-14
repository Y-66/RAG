from langchain_chroma import Chroma


def build_retriever(vector_store: Chroma, k: int = 1, search_type: str = "similarity"):
    """Return a retriever configured from a vector store."""

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

