"""
Query rewriting techniques: HyDE (Hypothetical Document Embedding) and PICO decomposition.
"""
from typing import List, Dict, Any, Optional, Tuple
import re
from ..generation.prompt_templates import SYSTEM_PROMPT, QUERY_TEMPLATE
from ..config import settings
from anthropic import Anthropic
import json


class QueryRewriter:
    def __init__(self, anthropic_client: Anthropic = None):
        """
        Initialize query rewriter.

        Args:
            anthropic_client: Anthropic client instance for LLM calls
        """
        self.anthropic_client = anthropic_client or Anthropic()
        self.model_name = settings.LLM_MODEL_NAME

    def hyde(self, query: str) -> str:
        """
        Generate Hypothetical Document Embedding (HyDE) for a query.

        Args:
            query: Original query string

        Returns:
            Hypothetical passage string that would answer the query
        """
        # Create prompt for HyDE
        hyde_prompt = f"""Write a short passage from a clinical research paper that would answer the following question:

        Question: {query}

        The passage should be written in the style of an abstract or results section from a biomedical paper,
        containing plausible details about study design, findings, and conclusions that directly address the question.
        Keep it concise but informative - approximately 2-3 sentences."""

        try:
            message = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=200,
                temperature=0.3,
                system="You are a biomedical research scientist writing hypothetical paper passages.",
                messages=[
                    {"role": "user", "content": hyde_prompt}
                ]
            )

            # Extract the generated text
            hypothetical_passage = message.content[0].text.strip()
            return hypothetical_passage
        except Exception as e:
            print(f"Error generating HyDE passage: {e}")
            # Fallback to original query
            return query

    def decompose_pico(self, query: str) -> Dict[str, Optional[str]]:
        """
        Decompose a query into PICO components (Population, Intervention, Comparator, Outcome).

        Args:
            query: Query string to decompose

        Returns:
            Dictionary with keys: population, intervention, comparator, outcome
        """
        # Use LLM to extract PICO components
        pico_prompt = f"""Extract the PICO components from the following clinical research question.
        Return your answer as a JSON object with exactly these keys: "population", "intervention", "comparator", "outcome".
        If a component is not present or unclear, set its value to null.

        Question: {query}

        Definitions:
        - Population: The patient group or demographic being studied (e.g., "elderly patients with type 2 diabetes")
        - Intervention: The treatment, exposure, or intervention being investigated (e.g., "metformin 500mg twice daily")
        - Comparator: What the intervention is being compared to (e.g., "placebo", "standard care", or null if not comparative)
        - Outcome: The clinical outcome or endpoint being measured (e.g., "liver enzyme levels", "mortality rate")

        Example:
        Question: "What are the cardiovascular risks of rosuvastatin in elderly patients?"
        Answer: {{"population": "elderly patients", "intervention": "rosuvastatin", "comparator": null, "outcome": "cardiovascular risks"}}

        Question: "Does metformin reduce liver toxicity compared to insulin in patients with hepatic impairment?"
        Answer: {{"population": "patients with hepatic impairment", "intervention": "metformin", "comparator": "insulin", "outcome": "liver toxicity"}}"""

        try:
            message = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=150,
                temperature=0.1,
                system="You are a clinical research expert specializing in PICO framework decomposition.",
                messages=[
                    {"role": "user", "content": pico_prompt}
                ]
            )

            # Extract and parse JSON response
            response_text = message.content[0].text.strip()

            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                pico_components = json.loads(json_str)

                # Ensure all required keys are present
                required_keys = ["population", "intervention", "comparator", "outcome"]
                for key in required_keys:
                    if key not in pico_components:
                        pico_components[key] = None

                return pico_components
            else:
                # Fallback: return all null if JSON parsing fails
                return {"population": None, "intervention": None, "comparator": None, "outcome": None}

        except Exception as e:
            print(f"Error decomposing PICO: {e}")
            # Return null components on error
            return {"population": None, "intervention": None, "comparator": None, "outcome": None}

    def expand_query_with_synonyms(self, query: str, drug_normalizer=None) -> List[str]:
        """
        Expand query with drug synonyms using RxNorm normalization.

        Args:
            query: Original query string
            drug_normalizer: DrugNormalizer instance for synonym expansion

        Returns:
            List of expanded query strings
        """
        if drug_normalizer is None:
            return [query]

        # Extract potential drug names from query (simple approach)
        # In practice, you'd use NER or more sophisticated extraction
        words = query.split()
        expanded_queries = [query]  # Start with original

        # Look for potential drug names (simplified - starts with capital or known drug patterns)
        for i, word in enumerate(words):
            # Clean word
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 2:
                # Try to normalize and get synonyms
                try:
                    synonyms = drug_normalizer.get_drug_synonyms(clean_word)
                    if synonyms and len(synonyms) > 1:  # Has synonyms beyond itself
                        # Create expanded queries with each synonym
                        for synonym in synonyms[:3]:  # Limit to top 3 synonyms
                            if synonym.lower() != clean_word.lower():
                                new_words = words.copy()
                                new_words[i] = synonym
                                expanded_query = ' '.join(new_words)
                                if expanded_query not in expanded_queries:
                                    expanded_queries.append(expanded_query)
                except Exception:
                    pass  # Skip if normalization fails

        return expanded_queries[:5]  # Limit total expanded queries

    def rewrite_query(self,
                     query: str,
                     use_hyde: bool = True,
                     use_pico: bool = True,
                     drug_normalizer=None) -> Dict[str, Any]:
        """
        Apply multiple query rewriting techniques.

        Args:
            query: Original query string
            use_hyde: Whether to apply HyDE
            use_pico: Whether to apply PICO decomposition
            drug_normalizer: DrugNormalizer instance for synonym expansion

        Returns:
            Dictionary containing original query and rewritten versions
        """
        result = {
            "original_query": query,
            "rewritten_queries": [query],  # Start with original
            "hyde_passage": None,
            "pico_components": None,
            "expanded_queries": []
        }

        # Apply PICO decomposition
        if use_pico:
            pico_components = self.decompose_pico(query)
            result["pico_components"] = pico_components

            # Create PICO-based query variations
            pico_parts = []
            if pico_components["population"]:
                pico_parts.append(pico_components["population"])
            if pico_components["intervention"]:
                pico_parts.append(pico_components["intervention"])
            if pico_components["comparator"]:
                pico_parts.append(pico_components["comparator"])
            if pico_components["outcome"]:
                pico_parts.append(pico_components["outcome"])

            if pico_parts:
                pico_query = " ".join(pico_parts)
                if pico_query not in result["rewritten_queries"]:
                    result["rewritten_queries"].append(pico_query)

        # Apply HyDE
        if use_hyde:
            hyde_passage = self.hyde(query)
            result["hyde_passage"] = hyde_passage
            # Add HyDE passage as a rewritten query for retrieval
            if hyde_passage and hyde_passage not in result["rewritten_queries"]:
                result["rewritten_queries"].append(hyde_passage)

        # Apply synonym expansion
        if drug_normalizer:
            expanded_queries = self.expand_query_with_synonyms(query, drug_normalizer)
            result["expanded_queries"] = expanded_queries
            # Add unique expanded queries to rewritten queries
            for exp_query in expanded_queries:
                if exp_query not in result["rewritten_queries"]:
                    result["rewritten_queries"].append(exp_query)

        # Limit total rewritten queries to avoid too many retrievals
        result["rewritten_queries"] = result["rewritten_queries"][:10]

        return result


# Convenience functions
def rewrite_query_hyde(query: str, anthropic_client: Anthropic = None) -> str:
    """
    Generate HyDE passage for a query.

    Args:
        query: Original query string
        anthropic_client: Anthropic client instance

    Returns:
        Hypothetical passage string
    """
    rewriter = QueryRewriter(anthropic_client)
    return rewriter.hyde(query)


def decompose_pico(query: str, anthropic_client: Anthropic = None) -> Dict[str, Optional[str]]:
    """
    Decompose query into PICO components.

    Args:
        query: Query string to decompose
        anthropic_client: Anthropic client instance

    Returns:
        Dictionary with PICO components
    """
    rewriter = QueryRewriter(anthropic_client)
    return rewriter.decompose_pico(query)


if __name__ == "__main__":
    # Test the query rewriter
    print("QueryRewriter initialized successfully")
    print("Requires Anthropic API key for full functionality")

    # Example usage
    test_query = "What are the known hepatotoxicity risks of metformin in elderly patients, and which clinical trials have studied this population?"
    print(f"\nTest query: {test_query}")

    # This would work with a valid Anthropic API key
    # rewriter = QueryRewriter()
    # pico_result = rewriter.decompose_pico(test_query)
    # print(f"PICO decomposition: {pico_result}")