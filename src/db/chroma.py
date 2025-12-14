import os
import shutil
from langchain_chroma import Chroma


def create_vector_store(
    persist_directory: str, collection_name: str, embedding_model
) -> Chroma:
    """Create (or load) a Chroma vector store with persistence."""

    # 清空旧向量库，确保使用最新分片
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )