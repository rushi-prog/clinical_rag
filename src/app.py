from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever

from src.retrieval.reranker import Reranker

from src.generation.rag_chain import (
    generate_answer
)


def load_retrieval_pipeline():

    vector_store = VectorStore(
        dimension=384
    )

    vector_store.load(
        "data/indexes/clinical_trials"
    )

    bm25_store = BM25Store()

    bm25_store.load(
        "data/indexes/clinical_trials"
    )

    retriever = HybridRetriever(
        vector_store,
        bm25_store
    )
    # print("Loading saved indexes...")
    return retriever


def main():

    retriever = (
        load_retrieval_pipeline()
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
        f"[Source {i+1}]\n{doc}"
        for i, (doc, score)
        in enumerate(reranked[:3])
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