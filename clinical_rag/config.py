from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: str
    NCBI_API_KEY: Optional[str] = None
    UMLS_API_KEY: Optional[str] = None
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None

    # Model names
    EMBEDDING_MODEL_NAME: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL_NAME: str = "claude-sonnet-4-20250514"

    # Chunking parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    PARENT_CHUNK_SIZE: int = 2048

    # Qdrant collection names
    BIOMEDICAL_CHUNKS_COLLECTION: str = "biomedical_chunks"
    BIOMEDICAL_PARENTS_COLLECTION: str = "biomedical_parents"

    # Retrieval parameters
    HYBRID_TOP_K: int = 100
    RERANKER_TOP_K: int = 5
    FINAL_TOP_K: int = 5

    # Rate limits (requests per second)
    PUBMED_RATE_LIMIT: float = 10.0  # with API key
    CLINICALTRIALS_RATE_LIMIT: float = 5.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()