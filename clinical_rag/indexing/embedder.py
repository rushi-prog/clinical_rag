"""
BioMedBERT embedding wrapper for text chunks.
"""
import numpy as np
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from ..config import settings


class BioMedEmbedder:
    def __init__(self, model_name: str = None):
        """
        Initialize BioMedBERT embedder.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading embedding model {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return np.array([]).reshape(0, self.embedding_dim)

        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string.

        Args:
            text: Text string to encode

        Returns:
            Numpy array of embedding with shape (1, embedding_dim)
        """
        if not text or not text.strip():
            return np.zeros((1, self.embedding_dim))

        embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding

    def embed_chunks(self, chunks: List[Dict[str, Any]], text_field: str = "text") -> List[Dict[str, Any]]:
        """
        Embed a list of chunk dictionaries and add embeddings to metadata.

        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing the text to embed

        Returns:
            List of chunk dictionaries with embeddings added to metadata
        """
        if not chunks:
            return chunks

        # Extract texts
        texts = [chunk.get(text_field, "") for chunk in chunks]

        # Generate embeddings
        embeddings = self.encode_texts(texts, show_progress=True)

        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            if i < len(embeddings):
                chunk["embedding"] = embeddings[i].tolist()  # Convert to list for JSON serialization
                chunk["metadata"]["embedding_model"] = self.model_name
                chunk["metadata"]["embedding_dim"] = self.embedding_dim

        return chunks


# Convenience functions
def embed_texts(texts: List[str], model_name: str = None) -> np.ndarray:
    """
    Embed a list of texts.

    Args:
        texts: List of text strings
        model_name: Name of the sentence-transformers model

    Returns:
        Numpy array of embeddings
    """
    embedder = BioMedEmbedder(model_name)
    return embedder.encode_texts(texts)


def embed_single_text(text: str, model_name: str = None) -> np.ndarray:
    """
    Embed a single text string.

    Args:
        text: Text string to embed
        model_name: Name of the sentence-transformers model

    Returns:
        Numpy array of embedding
    """
    embedder = BioMedEmbedder(model_name)
    return embedder.encode_single(text)


if __name__ == "__main__":
    # Test the embedder
    sample_texts = [
        "Metformin is associated with reduced liver enzyme levels in elderly patients.",
        "Clinical trials show metformin improves glycemic control in type 2 diabetes.",
        "Adverse drug reactions require careful monitoring in pharmacovigilance studies."
    ]

    embedder = BioMedEmbedder()
    embeddings = embedder.encode_texts(sample_texts)

    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")