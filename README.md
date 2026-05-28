# Clinical Trial Literature Synthesizer

A production-grade RAG system that answers pharmacovigilance and clinical trial design questions by retrieving and synthesizing evidence from PubMed, ClinicalTrials.gov, and FDA FAERS. Every answer is grounded with full citation chains (PMID / NCT IDs).

## Overview

This system implements a Retrieval-Augmented Generation (RAG) pipeline specifically designed for clinical trial and pharmacovigilance literature. It combines:

- **Multi-source data ingestion** from PubMed, ClinicalTrials.gov, and FDA FAERS
- **Advanced retrieval** using hybrid BM25 + dense vector search with RRF fusion
- **Cross-encoder reranking** for improved relevance
- **Query rewriting** with HyDE and PICO decomposition
- **Parent-child chunking** for better context preservation
- **Drug name normalization** via RxNorm
- **Attribution scoring** to detect and warn about potential hallucinations
- **Contradiction detection** to surface conflicting evidence
- **FastAPI interface** for easy integration

## Project Structure

```
clinical-rag/
├── ingestion/
│   ├── pubmed_fetcher.py        # PubMed API wrapper
│   ├── clinicaltrials_fetcher.py # ClinicalTrials.gov API wrapper
│   ├── faers_fetcher.py         # FDA FAERS OpenFDA API wrapper
│   ├── grobid_parser.py         # Section-aware PDF → structured chunks (placeholder)
│   └── normalizer.py            # UMLS/RxNorm drug name normalization
├── indexing/
│   ├── chunker.py               # Section-aware + parent-child chunking
│   ├── embedder.py              # BioMedBERT embedding wrapper
│   ├── qdrant_store.py          # Qdrant collection setup + upsert
│   └── bm25_store.py            # BM25 index build + persist
├── retrieval/
│   ├── hybrid_retriever.py      # RRF fusion of BM25 + dense results
│   ├── reranker.py              # Cross-encoder reranker, top-k
│   └── query_rewriter.py        # HyDE + PICO query decomposition
├── generation/
│   ├── prompt_templates.py      # System + user prompt templates
│   ├── chain.py                 # LCEL chain: retrieve → rerank → generate
│   └── attribution.py           # Post-gen attribution scoring per claim
├── evaluation/
│   ├── ragas_eval.py            # RAGAS faithfulness + relevancy pipeline
│   ├── retrieval_eval.py        # Recall@k, MRR on BioASQ eval set
│   └── contradiction_detector.py # Flag conflicting claims in retrieved chunks
├── api/
│   └── server.py                # FastAPI server exposing /query endpoint
├── config.py                    # Pydantic settings (API keys, model names)
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in a `.env` file:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   NCBI_API_KEY=your_ncbi_api_key  # Optional but recommended
   UMLS_API_KEY=your_umls_api_key  # Optional
   QDRANT_URL=http://localhost:6333  # For local Qdrant
   QDRANT_API_KEY=optional_qdrant_key
   ```

## Usage

### Quick Start

1. Start Qdrant (for local development):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. Start the FastAPI server:
   ```bash
   python -m uvicorn clinical-rag.api.server:app --reload
   ```

3. Send a query:
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{
              "question": "What are the known hepatotoxicity risks of metformin in elderly patients, and which clinical trials have studied this population?"
            }'
   ```

### As a Library

```python
from clinical-rag.generation.chain import create_rag_chain
from clinical-rag.indexing.qdrant_store import QdrantStore
from clinical-rag.indexing.bm25_store import BM25Store
from clinical-rag.indexing.embedder import BioMedEmbedder
from clinical-rag.retrieval.hybrid_retriever import HybridRetriever
from clinical-rag.retrieval.reranker import ReRanker

# Initialize components
qdrant_store = QdrantStore()
bm25_store = BM25Store()  # Would need to be built/loaded with data
embedder = BioMedEmbedder()
retriever = HybridRetriever(bm25_store=bm25_store, qdrant_store=qdrant_store)
reranker = ReRanker()

# Create RAG chain
rag_chain = create_rag_chain(
    retriever=retriever,
    reranker=reranker,
    anthropic_api_key="your_anthropic_api_key"
)

# Query the system
result = rag_chain.invoke({
    "question": "What are the liver effects of metformin in elderly patients?"
})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Attribution scores: {result['attribution_scores']}")
print(f"Warnings: {result['warnings']}")
```

## Key Features

### Faithfulness-Focused Generation
- Every factual claim must be cited with PMID or NCT ID
- System responds with "Insufficient evidence" rather than hallucinating
- Attribution scoring detects potentially unsupported claims

### Advanced Retrieval
- Hybrid search combining BM25 (keyword) and dense vector (semantic) search
- Reciprocal Rank Fusion (RRF) for robust score combination
- Cross-encoder reranking for improved precision
- Query rewriting with HyDE (Hypothetical Document Embedding)
- PICO decomposition for structured query understanding

### Clinical-Specific Optimizations
- Drug name normalization via RxNorm (aspirin = ASA = acetylsalicylic acid)
- Section-aware chunking preserves document structure
- Parent-child retrieval: precise embeddings with contextual generation
- Biomedical-aware tokenization for BM25
- Specialized handling of clinical trial data structures

### Evaluation & Safety
- RAGAS evaluation pipeline for faithfulness and relevancy
- Retrieval evaluation using BioASQ dataset
- Contradiction detection to surface conflicting evidence
- Comprehensive test suite planned

## Data Sources

### PubMed
- Fetcher uses Biopython Entrez API
- Retrieves abstracts, metadata, MeSH terms, publication types
- Supports bulk fetch with pagination

### ClinicalTrials.gov
- Uses v2 REST API
- Focuses on Interventional studies, Phases II-IV
- Extracts NCT ID, phase, status, conditions, interventions, outcomes

### FDA FAERS
- Uses OpenFDA API
- Aggregates adverse event reports by drug name
- Tracks serious outcomes (death, hospitalization, etc.)

## Configuration

All configuration is handled via `config.py` using Pydantic settings. Key parameters:

- `CHUNK_SIZE=512`: Child chunk size in tokens
- `CHUNK_OVERLAP=64`: Overlap between child chunks
- `PARENT_CHUNK_SIZE=2048`: Parent chunk size for context
- `HYBRID_TOP_K=100`: Number of results from hybrid search
- `RERANKER_TOP_K=5`: Final number after reranking
- `EMBEDDING_MODEL_NAME`: BioMedBERT model for embeddings
- `RERANKER_MODEL_NAME`: Cross-encoder model for reranking
- `LLM_MODEL_NAME`: Claude model for generation

## Next Steps / To-Do

1. **Data ingestion pipelines** - Connect to actual APIs and schedule regular updates
2. **GROBID integration** - For full-text PDF processing when abstracts aren't sufficient
3. **Evaluation benchmarks** - Create clinical-specific test sets
4. **Performance optimization** - Add caching, batch processing, async operations
5. **UI component** - Build a simple web interface for testing
6. **Deployment** - Dockerize for production deployment
7. **Monitoring** - Add logging, metrics, and health checks

## Acknowledgments

Built with:
- [LangChain](https://www.langchain.com/) for LCEL chains
- [Sentence Transformers](https://www.sbert.net/) for embeddings and reranking
- [Qdrant](https://qdrant.tech/) for vector storage
- [Anthropic Claude](https://www.anthropic.com/) for generation
- [BioASQ](http://bioasq.org/) for evaluation dataset

## License

MIT License - see LICENSE file for details.