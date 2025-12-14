from constants import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, DEFAULT_PDF_PATH
from db.chroma import create_vector_store
from loader.pdf_loader import load_pdf
from model_factory import get_embedding_model
from retriever.my_retriever import build_retriever
from splitter.recursive_splitter import split_documents


def build_pipeline(pdf_path: str = DEFAULT_PDF_PATH):
    """Load, split, embed, and return retriever + store for a given PDF."""

    # 1. Load documents from PDF
    docs = load_pdf(pdf_path)

    # 2. Split documents into smaller chunks
    splits = split_documents(docs)

    # 3. Get Embedding model and Create vector store
    embedding_model = get_embedding_model()
    vector_store = create_vector_store(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
        embedding_model=embedding_model,
    )

    # 4. Embed and add documents to vector store
    vector_store.add_documents(documents=splits)

    # 5. Build retriever from vector store
    retriever = build_retriever(vector_store, k=1)

    return retriever, vector_store, embedding_model, splits


def main():
    query = "其在数据分析方面的优势受益于 Python 丰富的生态扩展库"

    retriever, vector_store, embedding_model, splits = build_pipeline()

    print("----------------------数据分片----------------------------------------\n")
    print(f"Loaded {len(splits)} chunks from PDF\n")

    print("----------------------向量生成----------------------------------------\n")
    vector_1 = embedding_model.embed_query(splits[0].page_content)
    print(f"Generated vectors of length {len(vector_1)}\n")

    print("----------------------retriever检索结果-返回Document对象列表---------------------------------------\n")
    results = retriever.invoke(query)
    print(results)

    print("----------------------vector_store原生方法-相似度搜索结果----------------------------------------\n")
    results_with_score = vector_store.similarity_search_with_score(query, k=1)
    doc, score = results_with_score[0]
    print(f"Score: {score}\n")
    print(doc)


if __name__ == "__main__":
    main()