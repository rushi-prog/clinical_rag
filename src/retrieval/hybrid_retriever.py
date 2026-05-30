from collections import defaultdict

from src.retrieval.vector_store import VectorStore
from src.retrieval.embedder import embed_texts
from src.retrieval.bm25_store import BM25Store


class HybridRetriever:

    def __init__(
        self,
        vector_store,
        bm25_store
    ):

        self.vector_store = vector_store

        self.bm25_store = bm25_store

    def reciprocal_rank_fusion(
        self,
        faiss_docs,
        bm25_docs,
        k=60
    ):

        scores = defaultdict(float)

        for rank, doc in enumerate(
            faiss_docs,
            start=1
        ):

            scores[doc] += (
                1 / (k + rank)
            )

        for rank, (doc, _) in enumerate(
            bm25_docs,
            start=1
        ):

            scores[doc] += (
                1 / (k + rank)
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked

    def search(
        self,
        query,
        k=3
    ):

        query_embedding = (
            embed_texts([query])[0]
        )

        faiss_docs = (
            self.vector_store.search(
                query_embedding,
                k=k
            )
        )

        bm25_docs = (
            self.bm25_store.search(
                query,
                k=k
            )
        )

        return self.reciprocal_rank_fusion(
            faiss_docs,
            bm25_docs
        )