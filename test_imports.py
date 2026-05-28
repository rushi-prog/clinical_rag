"""
Test script to verify all modules can be imported successfully.
"""
import sys
import traceback

def test_import(module_name, description=""):
    """Test importing a module and report success/failure."""
    try:
        __import__(module_name)
        print(f"✓ {module_name} {description}")
        return True
    except Exception as e:
        print(f"✗ {module_name} {description}: {e}")
        traceback.print_exc()
        return False

def main():
    """Test all imports."""
    print("Testing imports for Clinical Trial Literature Synthesizer...\n")

    success_count = 0
    total_count = 0

    # Test config
    total_count += 1
    if test_import("clinical-rag.config", "- Configuration"):
        success_count += 1

    # Test ingestion modules
    total_count += 1
    if test_import("clinical-rag.ingestion.pubmed_fetcher", "- PubMed Fetcher"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.ingestion.clinicaltrials_fetcher", "- ClinicalTrials Fetcher"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.ingestion.faers_fetcher", "- FAERS Fetcher"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.ingestion.normalizer", "- Drug Normalizer"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.ingestion.grobid_parser", "- GROBID Parser"):
        success_count += 1

    # Test indexing modules
    total_count += 1
    if test_import("clinical-rag.indexing.chunker", "- Chunker"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.indexing.embedder", "- Embedder"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.indexing.qdrant_store", "- Qdrant Store"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.indexing.bm25_store", "- BM25 Store"):
        success_count += 1

    # Test retrieval modules
    total_count += 1
    if test_import("clinical-rag.retrieval.hybrid_retriever", "- Hybrid Retriever"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.retrieval.reranker", "- Reranker"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.retrieval.query_rewriter", "- Query Rewriter"):
        success_count += 1

    # Test generation modules
    total_count += 1
    if test_import("clinical-rag.generation.prompt_templates", "- Prompt Templates"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.generation.chain", "- RAG Chain"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.generation.attribution", "- Attribution Scorer"):
        success_count += 1

    # Test evaluation modules
    total_count += 1
    if test_import("clinical-rag.evaluation.ragas_eval", "- RAGAS Evaluator"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.evaluation.retrieval_eval", "- Retrieval Evaluator"):
        success_count += 1

    total_count += 1
    if test_import("clinical-rag.evaluation.contradiction_detector", "- Contradiction Detector"):
        success_count += 1

    # Test API module
    total_count += 1
    if test_import("clinical-rag.api.server", "- FastAPI Server"):
        success_count += 1

    print(f"\n{'='*50}")
    print(f"Import test results: {success_count}/{total_count} successful")

    if success_count == total_count:
        print("🎉 All imports successful!")
        return 0
    else:
        print(f"❌ {total_count - success_count} imports failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())