"""
Prompt templates for the RAG generation chain.
"""
from typing import Dict, Any


# System prompt for the clinical research assistant
SYSTEM_PROMPT = """You are a clinical research assistant. Answer the question using ONLY the provided context.
For every factual claim, cite the source using [PMID:XXXXXXXX] or [NCT:XXXXXXXXXX].
If the context does not contain enough information to answer, say "Insufficient evidence in retrieved sources."
Never speculate beyond what the sources store."""

# User prompt template
QUERY_TEMPLATE = """Context:
{context}

Question: {question}

Instructions: Answer in structured format. For each claim, add inline citation.
End with a "Sources" section listing all cited PMIDs and NCT IDs."""


def format_prompt(context: str, question: str) -> str:
    """
    Format the user prompt with context and question.

    Args:
        context: Retrieved context text
        question: User's question

    Returns:
        Formatted prompt string
    """
    return QUERY_TEMPLATE.format(context=context, question=question)


def get_system_prompt() -> str:
    """
    Get the system prompt for the LLM.

    Returns:
        System prompt string
    """
    return SYSTEM_PROMPT


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Args:
        chunks: List of retrieved chunk dictionaries

    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant sources found."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        # Create citation identifier
        citation_parts = []
        if metadata.get("pmid"):
            citation_parts.append(f"PMID:{metadata['pmid']}")
        if metadata.get("nct_id"):
            citation_parts.append(f"NCT:{metadata['nct_id']}")
        if metadata.get("year"):
            citation_parts.append(str(metadata["year"]))

        citation = " [" + ", ".join(citation_parts) + "]" if citation_parts else ""

        # Add source information
        source_info = ""
        if metadata.get("source"):
            source_info = f" [{metadata['source'].upper()}]"

        context_parts.append(f"[{i}]{source_info}{citation}: {text}")

    return "\n\n".join(context_parts)


def extract_citations(text: str) -> Dict[str, List[str]]:
    """
    Extract PMID and NCT IDs from text.

    Args:
        text: Text to search for citations

    Returns:
        Dictionary with keys 'pmids' and 'nct_ids' containing lists of found identifiers
    """
    import re

    pmids = re.findall(r'PMID:(\d+)', text, re.IGNORECASE)
    nct_ids = re.findall(r'NCT:(\d+)', text, re.IGNORECASE)

    return {
        "pmids": list(set(pmids)),  # Remove duplicates
        "nct_ids": list(set(nct_ids))
    }


if __name__ == "__main__":
    # Test the prompt templates
    sample_chunks = [
        {
            "text": "Metformin was associated with reduced ALT levels in elderly patients with type 2 diabetes.",
            "metadata": {
                "pmid": "12345678",
                "year": "2023",
                "source": "pubmed"
            }
        },
        {
            "text": "NCT01234567 studied metformin liver effects in geriatric population over 24 weeks.",
            "metadata": {
                "nct_id": "01234567",
                "year": "2022",
                "source": "clinicaltrials"
            }
        }
    ]

    context = format_context(sample_chunks)
    print("Formatted context:")
    print(context)
    print("\n" + "="*50 + "\n")

    prompt = format_prompt(context, "What are the liver effects of metformin in elderly patients?")
    print("Formatted prompt:")
    print(prompt)