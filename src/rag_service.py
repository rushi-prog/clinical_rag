from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.rag_chain import generate_answer


vector_store = VectorStore(dimension=384)
vector_store.load("data/indexes/clinical_trials")

bm25_store = BM25Store()
bm25_store.load("data/indexes/clinical_trials")

retriever = HybridRetriever(
    vector_store,
    bm25_store
)

reranker = Reranker()


def ask(query: str):

    retrieved = retriever.search(
        query,
        k=20
    )

    docs = [
        doc
        for doc, score in retrieved
    ]

    reranked = reranker.rerank(
        query,
        docs
    )

    top_docs = [
        f"[Source {i+1}]\n{doc}"
        for i, (doc, score)
        in enumerate(reranked[:5])
    ]

    answer = generate_answer(
        query,
        top_docs
    )

    return {
        "answer": answer,
        "sources": top_docs
    }