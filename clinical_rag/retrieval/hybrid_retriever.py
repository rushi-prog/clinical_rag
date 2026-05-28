"""
Hybrid retrieval combining BM25 and dense vector retrieval with RRF fusion.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..indexing.bm25_store import BM25Store
from ..indexing.qdrant_store import QdrantStore
from ..indexing.embedder import BioMedEmbedder
from ..config import settings


class HybridRetriever:
    def __init__(self,
                 bm25_store: BM25Store = None,
                 qdrant_store: QdrantStore = None,
                 embedder: BioMedEmbedder = None):
        """
        Initialize hybrid retriever.

        Args:
            bm25_store: BM25Store instance
            qdrant_store: QdrantStore instance
            embedder: BioMedEmbedder instance
        """
        self.bm25_store = bm25_store
        self.qdrant_store = qdrant_store
        self.embedder = embedder or BioMedEmbedder()

        # RRF parameter
        self.k = 60  # Default RRF parameter

    def _reciprocal_rank_fusion(self,
                               bm25_results: List[Tuple[Dict[str, Any], float]],
                               dense_results: List[Dict[str, Any]],
                               top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Fuse BM25 and dense retrieval results using Reciprocal Rank Fusion.

        Args:
            bm25_results: List of (chunk, score) tuples from BM25
            dense_results: List of chunk dictionaries with scores from dense retrieval
            top_k: Number of top results to return after fusion

        Returns:
            List of fused and ranked chunk dictionaries
        """
        # Create dictionaries to store RRF scores by chunk ID
        rrf_scores = {}
        chunk_map = {}  # Map chunk ID to chunk data

        # Process BM25 results
        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            # Use chunk text hash as ID for consistency
            chunk_id = hash(chunk.get("text", ""))  # Simple hash for demo
            rrf_score = 1.0 / (self.k + rank)

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0
                chunk_map[chunk_id] = chunk

            rrf_scores[chunk_id] += rrf_score

        # Process dense results
        for rank, chunk in enumerate(dense_results, start=1):
            # Use chunk text hash as ID for consistency
            chunk_id = hash(chunk.get("text", ""))
            rrf_score = 1.0 / (self.k + rank)

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0
                chunk_map[chunk_id] = chunk

            rrf_scores[chunk_id] += rrf_score

        # Sort by RRF score descending
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top_k chunks
        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            chunk = chunk_map[chunk_id].copy()
            chunk["metadata"]["rrf_score"] = score
            results.append(chunk)

        return results

    def retrieve(self,
                query: str,
                top_k: int = None,
                filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search with RRF fusion.

        Args:
            query: Query string
            top_k: Number of results to return
            filters: Optional metadata filters for Qdrant search

        Returns:
            List of retrieved chunk dictionaries
        """
        top_k = top_k or settings.HYBRID_TOP_K

        # Get BM25 results
        bm25_results = []
        if self.bm25_store:
            bm25_results = self.bm25_store.search(query, top_k=top_k)

        # Get dense results
        dense_results = []
        if self.qdrant_store:
            # Encode query
            query_embedding = self.embedder.encode_single(query)
            query_vector = query_embedding[0].tolist()  # Get first (and only) embedding

            # Search Qdrant
            dense_results = self.qdrant_store.search_chunks(
                query_vector=query_vector,
                limit=top_k,
                filter_conditions=filters
            )

        # Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
            top_k=top_k
        )

        return fused_results


# Convenience functions
def hybrid_retrieve(query: str,
                   bm25_store: BM25Store = None,
                   qdrant_store: QdrantStore = None,
                   top_k: int = None) -> List[Dict[str, Any]]:
    """
    Perform hybrid retrieval with RRF fusion.

    Args:
        query: Query string
        bm25_store: BM25Store instance
        qdrant_store: QdrantStore instance
        top_k: Number of results to return

    Returns:
        List of retrieved chunk dictionaries
    """
    retriever = HybridRetriever(bm25_store=bm25_store, qdrant_store=qdrant_store)
    return retriever.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    # Test the hybrid retriever (requires initialized stores)
    print("HybridRetriever initialized successfully")
    print("Requires BM25Store and QdrantStore instances to be functional")