from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever

from src.retrieval.reranker import Reranker

from src.generation.rag_chain import (
    generate_answer
)

from src.ingestion.data_loader import (
    load_trials
)


def build_retrieval_pipeline():

    df = load_trials(
        "data/clinical_trials.csv"
    )

    documents = list(
        df["abstract"]
    )

    embeddings = embed_texts(
        documents
    )

    vector_store = VectorStore(
        dimension=384
    )

    vector_store.add_documents(
        embeddings,
        documents
    )

    bm25_store = BM25Store()

    bm25_store.build(
        documents
    )

    retriever = HybridRetriever(
        vector_store,
        bm25_store
    )

    return retriever


def main():

    retriever = (
        build_retrieval_pipeline()
    )

    reranker = Reranker()

    query = input(
        "\nAsk a question: "
    )

    retrieved = retriever.search(
        query,
        k=5
    )

    docs = [
        doc
        for doc, score
        in retrieved
    ]

    reranked = reranker.rerank(
        query,
        docs
    )

    top_docs = [
        doc
        for doc, score
        in reranked[:3]
    ]

    answer = generate_answer(
        query,
        top_docs
    )

    print("\n")
    print("=" * 60)

    print("\nRetrieved Context:\n")

    for i, doc in enumerate(
        top_docs,
        start=1
    ):

        print(
            f"[{i}] {doc}\n"
        )

    print(
        "\nGenerated Answer:\n"
    )

    print(answer)

    print("\n" + "=" * 60)


if __name__ == "__main__":

    main()