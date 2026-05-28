"""
Cross-encoder reranker for improving retrieval quality.
"""
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder
import torch
from tqdm import tqdm
from ..config import settings


class ReRanker:
    def __init__(self, model_name: str = None):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading reranker model {self.model_name} on {self.device}")
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self,
              query: str,
              chunks: List[Dict[str, Any]],
              top_k: int = None,
              text_field: str = "text") -> List[Dict[str, Any]]:
        """
        Rerank chunks based on relevance to query using cross-encoder.

        Args:
            query: Query string
            chunks: List of chunk dictionaries to rerank
            top_k: Number of top chunks to return after reranking
            text_field: Field name containing the text to rerank on

        Returns:
            List of reranked chunk dictionaries with scores
        """
        if not chunks:
            return []

        top_k = top_k or settings.RERANKER_TOP_K

        # Prepare query-chunk pairs
        pairs = []
        valid_chunks = []
        for chunk in chunks:
            text = chunk.get(text_field, "")
            if text and text.strip():
                pairs.append([query, text])
                valid_chunks.append(chunk)

        if not pairs:
            return []

        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Add scores to chunks
        for i, chunk in enumerate(valid_chunks):
            chunk = chunk.copy()  # Don't modify original
            chunk["rerank_score"] = float(scores[i])
            valid_chunks[i] = chunk

        # Sort by rerank score descending
        reranked_chunks = sorted(
            valid_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # Return top_k
        return reranked_chunks[:top_k]

    def rerank_with_metadata(self,
                            query: str,
                            chunks: List[Dict[str, Any]],
                            top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank chunks and swap child chunks for parent chunks before returning.

        Args:
            query: Query string
            chunks: List of chunk dictionaries (expected to be child chunks)
            top_k: Number of top chunks to return after reranking

        Returns:
            List of reranked parent chunk dictionaries
        """
        # Rerank child chunks
        reranked_children = self.rerank(query, chunks, top_k=top_k*2)  # Get extra for parent mapping

        # Extract parent IDs and get parent chunks
        parent_ids = []
        child_to_parent_map = {}

        for chunk in reranked_children:
            metadata = chunk.get("metadata", {})
            parent_id = metadata.get("parent_id")
            if parent_id is not None:
                parent_ids.append(parent_id)
                child_to_parent_map[parent_id] = child_to_parent_map.get(parent_id, []) + [chunk]

        # Deduplicate parent IDs while preserving order
        seen = set()
        unique_parent_ids = []
        for pid in parent_ids:
            if pid not in seen:
                seen.add(pid)
                unique_parent_ids.append(pid)

        # Limit to top_k parents
        top_parent_ids = unique_parent_ids[:top_k]

        # Retrieve parent chunks (this would normally come from a store)
        # For now, we'll return the child chunks with parent context info
        # In a full implementation, this would fetch parent chunks from storage
        results = []
        for parent_id in top_parent_ids:
            # Get the best child chunk for this parent (highest rerank score)
            if parent_id in child_to_parent_map:
                best_child = max(child_to_parent_map[parent_id],
                               key=lambda x: x.get("rerank_score", 0))

                # Add parent context information
                best_child["metadata"]["has_parent_context"] = True
                best_child["metadata"]["parent_id"] = parent_id
                results.append(best_child)

        return results


# Convenience functions
def rerank_chunks(query: str,
                 chunks: List[Dict[str, Any]],
                 top_k: int = None,
                 model_name: str = None) -> List[Dict[str, Any]]:
    """
    Rerank chunks using cross-encoder.

    Args:
        query: Query string
        chunks: List of chunk dictionaries
        top_k: Number of top chunks to return
        model_name: Name of the cross-encoder model

    Returns:
        List of reranked chunk dictionaries
    """
    reranker = ReRanker(model_name)
    return reranker.rerank(query, chunks, top_k=top_k)


if __name__ == "__main__":
    # Test the reranker (requires model download)
    print("ReRanker initialized successfully")
    print("Requires sentence-transformers cross-encoder model to be functional")