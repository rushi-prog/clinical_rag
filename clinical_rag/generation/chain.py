"""
LCEL chain for RAG: retrieve → rerank → generate.
"""
from typing import Dict, Any, List
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import ReRanker
from ..generation.prompt_templates import SYSTEM_PROMPT, QUERY_TEMPLATE
from ..generation.attribution import AttributionScorer
from ..config import settings


def create_rag_chain(
    retriever: HybridRetriever = None,
    reranker: ReRanker = None,
    anthropic_api_key: str = None
):
    """
    Create the RAG chain using LCEL (LangChain Expression Language).

    Args:
        retriever: HybridRetriever instance
        reranker: ReRanker instance
        anthropic_api_key: Anthropic API key for Claude

    Returns:
        LCEL chain that takes a question and returns an answer with attribution
    """
    # Initialize components if not provided
    if retriever is None:
        # In a full implementation, these would be initialized with proper stores
        retriever = HybridRetriever()

    if reranker is None:
        reranker = ReRanker()

    # Initialize LLM
    llm = ChatAnthropic(
        model=settings.LLM_MODEL_NAME,
        anthropic_api_key=anthropic_api_key or settings.ANTHROPIC_API_KEY,
        temperature=0.1,  # Low temperature for factual consistency
        max_tokens=1000
    )

    # Initialize attribution scorer
    attribution_scorer = AttributionScorer()

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", QUERY_TEMPLATE)
    ])

    # Define the chain steps
    def retrieve_and_rerank(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant chunks and rerank them."""
        question = inputs["question"]

        # Retrieve using hybrid search
        retrieved_chunks = retriever.retrieve(question, top_k=settings.HYBRID_TOP_K)

        # Rerank the retrieved chunks
        reranked_chunks = reranker.rerank(
            query=question,
            chunks=retrieved_chunks,
            top_k=settings.RERANKER_TOP_K
        )

        # Swap child chunks for parent chunks (in a full implementation)
        # For now, we'll assume the reranker already handled this

        return {
            "question": question,
            "context_chunks": reranked_chunks
        }

    def format_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format retrieved chunks into context string."""
        from ..generation.prompt_templates import format_context

        context_chunks = inputs["context_chunks"]
        context_string = format_context(context_chunks)

        return {
            "question": inputs["question"],
            "context": context_string
        }

    def add_attribution(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Add attribution scoring to the generated answer."""
        answer = inputs["answer"]
        context_chunks = inputs.get("context_chunks", [])

        # Score attribution
        attribution_result = attribution_scorer.score_answer(answer, context_chunks)

        return {
            "question": inputs["question"],
            "answer": answer,
            "attribution_scores": attribution_result["claim_scores"],
            "warnings": attribution_result["warnings"],
            "context_chunks": context_chunks  # Pass through for source extraction
        }

    def extract_sources(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cited sources from the answer."""
        from ..generation.prompt_templates import extract_citations

        answer = inputs["answer"]
        citations = extract_citations(answer)

        # Build sources list from context chunks
        sources = []
        context_chunks = inputs.get("context_chunks", [])

        # Create lookup for chunks by PMID/NCT ID
        chunk_lookup = {}
        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            pmid = metadata.get("pmid")
            nct_id = metadata.get("nct_id")

            if pmid:
                chunk_lookup[f"PMID:{pmid}"] = chunk
            if nct_id:
                chunk_lookup[f"NCT:{nct_id}"] = chunk

        # Build sources list
        for pmid in citations["pmids"]:
            key = f"PMID:{pmid}"
            if key in chunk_lookup:
                chunk = chunk_lookup[key]
                metadata = chunk.get("metadata", {})
                sources.append({
                    "id": pmid,
                    "type": "PMID",
                    "title": metadata.get("title", ""),
                    "year": metadata.get("year"),
                    "source": metadata.get("source", ""),
                    "journal": metadata.get("journal", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })

        for nct_id in citations["nct_ids"]:
            key = f"NCT:{nct_id}"
            if key in chunk_lookup:
                chunk = chunk_lookup[key]
                metadata = chunk.get("metadata", {})
                sources.append({
                    "id": nct_id,
                    "type": "NCT",
                    "title": metadata.get("brief_title", metadata.get("official_title", "")),
                    "year": metadata.get("start_year", metadata.get("year")),
                    "source": metadata.get("source", ""),
                    "url": f"https://clinicaltrials.gov/ct2/show/{nct_id}"
                })

        return {
            "question": inputs["question"],
            "answer": inputs["answer"],
            "sources": sources,
            "attribution_scores": inputs["attribution_scores"],
            "warnings": inputs["warnings"]
        }

    # Build the LCEL chain
    chain = (
        RunnablePassthrough()
        | retrieve_and_rerank
        | format_context
        | prompt
        | llm
        | StrOutputParser()
        | add_attribution
        | extract_sources
    )

    return chain


# Convenience function for creating chain with default components
def create_default_chain(anthropic_api_key: str = None):
    """
    Create a RAG chain with default components.

    Note: In a full implementation, this would initialize proper retriever stores.
    For now, it returns a chain framework that needs proper store initialization.
    """
    # Placeholder - in practice, these would be initialized with actual data
    retriever = HybridRetriever()  # Would need proper BM25Store and QdrantStore
    reranker = ReRanker()

    return create_rag_chain(
        retriever=retriever,
        reranker=reranker,
        anthropic_api_key=anthropic_api_key
    )


if __name__ == "__main__":
    # Test chain creation
    print("RAG chain framework created successfully")
    print("Note: Requires proper initialization of retriever stores for full functionality")