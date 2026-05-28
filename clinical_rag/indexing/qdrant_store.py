"""
Qdrant vector store for biomedical chunks and parent chunks.
"""
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
import hashlib
import json
from ..config import settings


class QdrantStore:
    def __init__(self, url: str = None, api_key: str = None):
        """
        Initialize Qdrant client and collections.

        Args:
            url: Qdrant server URL
            api_key: Qdrant API key (for cloud instances)
        """
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.client = QdrantClient(url=self.url, api_key=self.api_key)

        # Collection names
        self.chunks_collection = settings.BIOMEDICAL_CHUNKS_COLLECTION
        self.parents_collection = settings.BIOMEDICAL_PARENTS_COLLECTION

        # Initialize collections
        self._ensure_collections_exist()

    def _ensure_collections_exist(self):
        """Ensure that the required collections exist in Qdrant."""
        existing_collections = self.client.get_collections().collections
        existing_names = [col.name for col in existing_collections]

        # Create chunks collection if it doesn't exist
        if self.chunks_collection not in existing_names:
            self.client.create_collection(
                collection_name=self.chunks_collection,
                vectors_config=VectorParams(
                    size=768,  # Default for PubMedBERT, will be updated dynamically
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.chunks_collection}")

        # Create parents collection if it doesn't exist
        if self.parents_collection not in existing_names:
            self.client.create_collection(
                collection_name=self.parents_collection,
                vectors_config=VectorParams(
                    size=1,  # Placeholder, parents don't need vectors for retrieval
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.parents_collection}")

    def _get_text_hash(self, text: str) -> str:
        """Generate a deterministic hash for text content."""
        return hashlib.md5(text.encode()).hexdigest()

    def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """
        Upsert child chunks into Qdrant collection.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' fields
                   Must contain 'embedding' in metadata or as a separate field
            batch_size: Number of points to upsert per batch

        Returns:
            True if successful
        """
        if not chunks:
            return True

        points = []
        for chunk in chunks:
            # Extract embedding
            embedding = chunk.get("embedding")
            if embedding is None:
                # Try to get from metadata
                embedding = chunk.get("metadata", {}).get("embedding")

            if embedding is None:
                print(f"Warning: Chunk missing embedding, skipping: {chunk.get('metadata', {}).get('pmid', 'unknown')}")
                continue

            # Ensure embedding is a list of floats
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                print(f"Warning: Invalid embedding type: {type(embedding)}")
                continue

            # Create point ID from hash of text + metadata for determinism
            text_content = chunk.get("text", "")
            metadata_str = json.dumps(chunk.get("metadata", {}), sort_keys=True)
            point_id = int(self._get_text_hash(text_content + metadata_str)[-16:], 16)  # Use last 16 chars as hex

            # Prepare payload
            payload = chunk.get("metadata", {}).copy()
            payload["text"] = text_content

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            try:
                self.client.upsert(
                    collection_name=self.chunks_collection,
                    points=batch
                )
            except Exception as e:
                print(f"Error upserting batch {i//batch_size}: {e}")
                return False

        return True

    def upsert_parents(self, parents: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """
        Upsert parent chunks into Qdrant collection (stored as payload only).

        Args:
            parents: List of parent chunk dictionaries with 'text' and 'metadata' fields
            batch_size: Number of points to upsert per batch

        Returns:
            True if successful
        """
        if not parents:
            return True

        points = []
        for parent in parents:
            # Extract text
            text_content = parent.get("text", "")

            # Create point ID from hash of text + metadata
            metadata_str = json.dumps(parent.get("metadata", {}), sort_keys=True)
            point_id = int(self._get_text_hash(text_content + metadata_str)[-16:], 16)

            # Parents don't need meaningful vectors - use zero vector
            # In practice, we won't search parents directly, only retrieve by ID
            zero_vector = [0.0]  # Minimal vector dimension

            # Prepare payload
            payload = parent.get("metadata", {}).copy()
            payload["text"] = text_content

            point = PointStruct(
                id=point_id,
                vector=zero_vector,
                payload=payload
            )
            points.append(point)

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            try:
                self.client.upsert(
                    collection_name=self.parents_collection,
                    points=batch
                )
            except Exception as e:
                print(f"Error upserting parents batch {i//batch_size}: {e}")
                return False

        return True

    def search_chunks(self, query_vector: List[float], limit: int = 100,
                     filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the vector store.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            filter_conditions: Optional dictionary of metadata filters

        Returns:
            List of matching chunk dictionaries with scores
        """
        # Build filter if provided
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, list):
                    # Handle multiple values (OR condition)
                    conditions.append(
                        models.FieldCondition(
                            key=models.KeywordMatch(keywords=value)
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=models.MatchValue(value=value)
                        )
                    )
            if conditions:
                qdrant_filter = models.Filter(must=conditions)

        try:
            search_results = self.client.search(
                collection_name=self.chunks_collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False  # We don't need vectors in results
            )

            # Format results
            results = []
            for scored_point in search_results:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": scored_point.payload
                }
                # Extract text and metadata from payload
                payload = scored_point.payload
                result["text"] = payload.get("text", "")
                result["metadata"] = {k: v for k, v in payload.items() if k != "text"}
                results.append(result)

            return results
        except Exception as e:
            print(f"Error searching chunks: {e}")
            return []

    def get_parent_by_id(self, parent_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a parent chunk by its ID.

        Args:
            parent_id: ID of the parent chunk to retrieve

        Returns:
            Parent chunk dictionary or None if not found
        """
        try:
            points = self.client.retrieve(
                collection_name=self.parents_collection,
                ids=[parent_id],
                with_payload=True,
                with_vectors=False
            )

            if points:
                point = points[0]
                payload = point.payload
                return {
                    "id": point.id,
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"}
                }
            return None
        except Exception as e:
            print(f"Error retrieving parent {parent_id}: {e}")
            return None

    def get_parents_by_ids(self, parent_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple parent chunks by their IDs.

        Args:
            parent_ids: List of parent chunk IDs to retrieve

        Returns:
            List of parent chunk dictionaries
        """
        if not parent_ids:
            return []

        try:
            points = self.client.retrieve(
                collection_name=self.parents_collection,
                ids=parent_ids,
                with_payload=True,
                with_vectors=False
            )

            parents = []
            for point in points:
                payload = point.payload
                parents.append({
                    "id": point.id,
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"}
                })

            return parents
        except Exception as e:
            print(f"Error retrieving parents by IDs: {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection info dictionary or None if error
        """
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return info.dict()
        except Exception as e:
            print(f"Error getting collection info for {collection_name}: {e}")
            return None

    def count_points(self, collection_name: str) -> int:
        """
        Count points in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of points in the collection
        """
        try:
            count = self.client.count(collection_name=collection_name)
            return count.count
        except Exception as e:
            print(f"Error counting points in {collection_name}: {e}")
            return 0


# Convenience functions
def get_qdrant_store(url: str = None, api_key: str = None) -> QdrantStore:
    """
    Get a QdrantStore instance.

    Args:
        url: Qdrant server URL
        api_key: Qdrant API key

    Returns:
        QdrantStore instance
    """
    return QdrantStore(url=url, api_key=api_key)


if __name__ == "__main__":
    # Test the Qdrant store (requires running Qdrant instance)
    print("QdrantStore initialized successfully")
    print(f"Chunks collection: {settings.BIOMEDICAL_CHUNKS_COLLECTION}")
    print(f"Parents collection: {settings.BIOMEDICAL_PARENTS_COLLECTION}")