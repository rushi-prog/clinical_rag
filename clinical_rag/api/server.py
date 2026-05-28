"""
FastAPI server for the Clinical Trial Literature Synthesizer RAG system.
"""
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from ..generation.chain import create_rag_chain
from ..config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Trial Literature Synthesizer",
    description="A production-grade RAG system for pharmacovigilance and clinical trial design questions",
    version="1.0.0"
)

# Global RAG chain (will be initialized on startup)
rag_chain = None


# Request and response models
class QueryRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    attribution_scores: Dict[int, float]
    warnings: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    message: str


class StatsResponse(BaseModel):
    index_size: int
    last_updated: Optional[str]
    collections: Dict[str, int]


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG chain on application startup."""
    global rag_chain
    try:
        logger.info("Initializing RAG chain...")
        # In a full implementation, this would initialize proper retriever stores
        # For now, we'll create a placeholder chain
        rag_chain = create_rag_chain()
        logger.info("RAG chain initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        # Don't raise - allow API to start but endpoints will return errors


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Clinical Trial Literature Synthesizer")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        message="Clinical Trial Literature Synthesizer is running"
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the vector store and indices."""
    # Placeholder implementation - in practice, this would query actual stores
    return StatsResponse(
        index_size=0,  # Would get from Qdrant store
        last_updated=None,
        collections={
            "biomedical_chunks": 0,
            "biomedical_parents": 0
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint for the RAG system.

    Args:
        request: QueryRequest containing question and optional filters

    Returns:
        QueryResponse with answer, sources, attribution scores, and warnings
    """
    global rag_chain

    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    try:
        logger.info(f"Processing query: {request.question}")

        # Invoke the RAG chain
        result = rag_chain.invoke({
            "question": request.question
        })

        # Extract results from chain output
        answer = result.get("answer", "Error generating answer")
        sources = result.get("sources", [])
        attribution_scores = result.get("attribution_scores", {})
        warnings = result.get("warnings", [])

        logger.info(f"Query processed successfully. Answer length: {len(answer)}")

        return QueryResponse(
            answer=answer,
            sources=sources,
            attribution_scores=attribution_scores,
            warnings=warnings
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Additional endpoints for data management (optional)
@app.post("/ingest/pubmed")
async def ingest_pubmed(query: str, max_results: int = 1000):
    """Ingest PubMed data (placeholder endpoint)."""
    # In a full implementation, this would trigger PubMed ingestion
    return {"message": f"Ingestion of PubMed data for query '{query}' initiated"}


@app.post("/ingest/clinicaltrials")
async def ingest_clinicaltrials(query: str = "", max_results: int = 1000):
    """Ingest ClinicalTrials.gov data (placeholder endpoint)."""
    # In a full implementation, this would trigger ClinicalTrials.gov ingestion
    return {"message": f"Ingestion of ClinicalTrials.gov data initiated"}


@app.post("/ingest/faers")
async def ingest_faers(drug_name: str, max_results: int = 1000):
    """Ingest FAERS data (placeholder endpoint)."""
    # In a full implementation, this would trigger FAERS ingestion
    return {"message": f"Ingestion of FAERS data for drug '{drug_name}' initiated"}


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )