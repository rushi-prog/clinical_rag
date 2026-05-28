"""
BM25 sparse retrieval index for biomedical chunks.
"""
import pickle
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
from ..config import settings


class BM25Store:
    def __init__(self, index_path: str = "bm25_index.pkl"):
        """
        Initialize BM25 store.

        Args:
            index_path: Path to save/load the BM25 index
        """
        self.index_path = index_path
        self.bm25 = None
        self.chunks = []  # Store original chunks for retrieval
        self.tokenized_chunks = []  # Store tokenized chunks for BM25

        # Biomedical-aware tokenizer parameters
        self.stopwords = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
            'very', 'when', 'come', 'may', 'will', 'on', 'than', 'that', 'them',
            'well', 'were'
        ])

        # Keep important biomedical terms that might be mistaken as stopwords
        self.keep_terms = set([
            'drug', 'dose', 'patient', 'treatment', 'study', 'trial', 'effect',
            'risk', 'benefit', 'adverse', 'event', 'outcome', 'response', 'therapy',
            'disease', 'syndrome', 'diagnosis', 'prognosis', 'metformin', 'insulin',
            'aspirin', 'ibuprofen', 'atorvastatin', 'lisinopril', 'warfarin'
        ])

    def _tokenize_biomedical(self, text: str) -> List[str]:
        """
        Tokenize text with biomedical-aware preprocessing.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Keep alphanumeric characters and hyphens (important for drug names)
        text = re.sub(r'[^a-z0-9\- ]', ' ', text)

        # Split on whitespace
        tokens = text.split()

        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Keep if it's not a stopword OR if it's in our keep list
            if token not in self.stopwords or token in self.keep_terms:
                # Further filter: remove very short tokens (except important ones like COX-2)
                if len(token) >= 2 or token in self.keep_terms:
                    filtered_tokens.append(token)

        return filtered_tokens

    def build_index(self, chunks: List[Dict[str, Any]], text_field: str = "text") -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing the text to index
        """
        self.chunks = chunks

        # Tokenize all chunks
        self.tokenized_chunks = []
        for chunk in tqdm(chunks, desc="Tokenizing chunks for BM25"):
            text = chunk.get(text_field, "")
            tokens = self._tokenize_biomedical(text)
            self.tokenized_chunks.append(tokens)

        # Create BM25 index
        if self.tokenized_chunks:
            self.bm25 = BM25Okapi(self.tokenized_chunks)
            print(f"Built BM25 index with {len(self.tokenized_chunks)} documents")
        else:
            self.bm25 = None
            print("Warning: No chunks to index for BM25")

    def save_index(self) -> None:
        """Save the BM25 index and chunks to disk."""
        if self.bm25 is None:
            print("Warning: No index to save")
            return

        data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_chunks': self.tokenized_chunks
        }

        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved BM25 index to {self.index_path}")
        except Exception as e:
            print(f"Error saving BM25 index: {e}")

    def load_index(self) -> bool:
        """
        Load the BM25 index and chunks from disk.

        Returns:
            True if successful
        """
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)

            self.bm25 = data['bm25']
            self.chunks = data['chunks']
            self.tokenized_chunks = data['tokenized_chunks']
            print(f"Loaded BM25 index from {self.index_path} with {len(self.chunks)} documents")
            return True
        except FileNotFoundError:
            print(f"BM25 index file not found: {self.index_path}")
            return False
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 100) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the BM25 index for relevant chunks.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            List of tuples (chunk, score) sorted by score descending
        """
        if self.bm25 is None:
            print("Warning: BM25 index not built or loaded")
            return []

        # Tokenize query
        query_tokens = self._tokenize_biomedical(query)
        if not query_tokens:
            return []

        # Get scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return chunks with scores
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                score = float(scores[idx])
                results.append((chunk, score))

        return results

    def search_ids_only(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Search and return only chunk IDs and scores.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            List of tuples (chunk_index, score)
        """
        if self.bm25 is None:
            return []

        query_tokens = self._tokenize_biomedical(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(int(idx), float(scores[idx])) for idx in top_indices if idx < len(self.chunks)]
        return results


# Convenience functions
def build_bm25_index(chunks: List[Dict[str, Any]], index_path: str = "bm25_index.pkl") -> BM25Store:
    """
    Build and save a BM25 index from chunks.

    Args:
        chunks: List of chunk dictionaries
        index_path: Path to save the index

    Returns:
        BM25Store instance
    """
    store = BM25Store(index_path)
    store.build_index(chunks)
    store.save_index()
    return store


def load_bm25_index(index_path: str = "bm25_index.pkl") -> Optional[BM25Store]:
    """
    Load a BM25 index from disk.

    Args:
        index_path: Path to the saved index

    Returns:
        BM25Store instance or None if failed
    """
    store = BM25Store(index_path)
    if store.load_index():
        return store
    return None


if __name__ == "__main__":
    # Test the BM25 store
    sample_chunks = [
        {"text": "Metformin reduces liver enzyme levels in elderly patients with diabetes.", "pmid": "12345"},
        {"text": "Clinical trials show ibuprofen increases risk of gastrointestinal bleeding.", "pmid": "67890"},
        {"text": "Atorvastatin is effective for lowering cholesterol in cardiovascular patients.", "pmid": "11111"},
        {"text": "Metformin-associated lactic acidosis is rare but serious in renal impairment.", "pmid": "22222"}
    ]

    # Build index
    bm25_store = build_bm25_index(sample_chunks, "test_bm25_index.pkl")

    # Search
    query = "metformin liver toxicity"
    results = bm25_store.search(query, top_k=2)

    print(f"\nSearch results for '{query}':")
    for chunk, score in results:
        print(f"  Score: {score:.4f} - {chunk['text'][:100]}...")

    # Test saving/loading
    bm25_store2 = load_bm25_index("test_bm25_index.pkl")
    if bmue5_store2:
        results2 = bm25_store2.search(query, top_k=2)
        print(f"\nLoaded index results for '{query}':")
        for chunk, score in results2:
            print(f"  Score: {score:.4f} - {chunk['text'][:100]}...")