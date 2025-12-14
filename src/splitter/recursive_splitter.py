from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    docs: list[Document], chunk_size: int = 250, chunk_overlap: int = 50
) -> list[Document]:
    """Split documents into smaller chunks for embedding."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return text_splitter.split_documents(docs)