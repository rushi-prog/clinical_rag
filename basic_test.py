"""
Basic functionality test for Clinical Trial Literature Synthesizer.
Tests core components without requiring external services.
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from clinical_rag.config import settings
        print(f"  Embedding model: {settings.EMBEDDING_MODEL_NAME}")
        print(f"  LLM model: {settings.LLM_MODEL_NAME}")
        print(f"  Chunk size: {settings.CHUNK_SIZE}")
        print("  [OK] Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] Configuration test failed: {e}")
        return False

def test_prompt_templates():
    """Test prompt templates."""
    print("\nTesting prompt templates...")
    try:
        from clinical_rag.generation.prompt_templates import (
            SYSTEM_PROMPT, QUERY_TEMPLATE, format_prompt, format_context
        )

        # Test system prompt
        assert "clinical research assistant" in SYSTEM_PROMPT.lower()
        assert "[PMID:" in SYSTEM_PROMPT

        # Test query template
        test_context = "Test context"
        test_question = "Test question?"
        formatted = format_prompt(test_context, test_question)
        assert test_context in formatted
        assert test_question in formatted

        # Test context formatting
        test_chunks = [
            {
                "text": "Metformin reduces liver enzymes.",
                "metadata": {
                    "pmid": "12345678",
                    "year": "2023",
                    "source": "pubmed"
                }
            }
        ]
        context_str = format_context(test_chunks)
        assert "Metformin reduces liver enzymes." in context_str
        assert "PMID:12345678" in context_str

        print("  [OK] Prompt templates work correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] Prompt template test failed: {e}")
        return False

def test_normalizer_basic():
    """Test drug normalizer basic functionality."""
    print("\nTesting drug normalizer...")
    try:
        from clinical_rag.ingestion.normalizer import normalize_drug_name

        # Test known mappings
        assert normalize_drug_name("asa") == "Aspirin"
        assert normalize_drug_name("advil") == "Ibuprofen"
        assert normalize_drug_name("lipitor") == "Atorvastatin"

        # Test case insensitivity
        assert normalize_drug_name("ASPIRIN") == "Aspirin"

        # Test unknown drug (should return title case)
        result = normalize_drug_name("unknown_drug")
        assert result == "Unknown_Drug" or result == "Unknown drug"

        print("  [OK] Drug normalizer works correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] Drug normalizer test failed: {e}")
        return False

def test_prompt_template_imports():
    """Test that we can import all the main components."""
    print("\nTesting component imports...")
    components = [
        ("clinical_rag.ingestion.pubmed_fetcher", "PubMedFetcher"),
        ("clinical_rag.ingestion.clinicaltrials_fetcher", "ClinicalTrialsFetcher"),
        ("clinical_rag.ingestion.faers_fetcher", "FAERSFetcher"),
        ("clinical_rag.indexing.chunker", "SectionAwareChunker"),
        ("clinical_rag.indexing.embedder", "BioMedEmbedder"),
        ("clinical_rag.retrieval.hybrid_retriever", "HybridRetriever"),
        ("clinical_rag.retrieval.reranker", "ReRanker"),
        ("clinical_rag.generation.chain", "create_rag_chain"),
        ("clinical_rag.evaluation.ragas_eval", "RAGEvaluator"),
    ]

    success_count = 0
    for module_path, class_name in components:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  [OK] {class_name}")
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] {class_name}: {e}")

    print(f"  Imported {success_count}/{len(components)} components")
    return success_count == len(components)

def main():
    """Run all basic tests."""
    print("=" * 60)
    print("Clinical Trial Literature Synthesizer - Basic Tests")
    print("=" * 60)

    tests = [
        test_config,
        test_prompt_templates,
        test_normalizer_basic,
        test_prompt_template_imports,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if test_func():
            passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} test groups passed")

    if passed == total:
        print("[OK] All basic tests passed!")
        print("\nThe system is ready for:")
        print("1. Installing dependencies: pip install -r requirements.txt")
        print("2. Setting up environment variables (.env file)")
        print("3. Starting required services (Qdrant, optionally GROBID)")
        print("4. Running data ingestion pipelines")
        print("5. Testing with the example query:")
        print('   > "What are the known hepatotoxicity risks of metformin in elderly patients, and which clinical trials have studied this population?"')
        return 0
    else:
        print("[FAIL] Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())