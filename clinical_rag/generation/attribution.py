"""
Attribution scoring for generated claims to detect hallucinations.
"""
from typing import List, Dict, Any, Tuple
import re
from sentence_transformers import CrossEncoder
import torch
import numpy as np
from tqdm import tqdm
from ..config import settings


class AttributionScorer:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        """
        Initialize attribution scorer with NLI model.

        Args:
            model_name: Name of the cross-encoder model for NLI (Natural Language Inference)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading attribution model {self.model_name} on {self.device}")
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as e:
            print(f"Warning: Could not load NLI model {model_name}: {e}")
            print("Falling back to simpler heuristic attribution scoring")
            self.model = None

    def _split_into_claims(self, text: str) -> List[str]:
        """
        Split text into individual claims for attribution scoring.

        Args:
            text: Generated answer text

        Returns:
            List of claim sentences
        """
        # Simple sentence splitting - in practice, use more sophisticated segmentation
        sentences = re.split(r'[.!?]+', text)
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                claims.append(sentence)
        return claims

    def _extract_citations_from_claim(self, claim: str) -> Tuple[List[str], List[str]]:
        """
        Extract PMID and NCT IDs from a claim.

        Args:
            claim: Claim sentence

        Returns:
            Tuple of (pmid_list, nct_id_list)
        """
        pmids = re.findall(r'PMID:(\d+)', claim, re.IGNORECASE)
        nct_ids = re.findall(r'NCT:(\d+)', claim, re.IGNORECASE)
        return pmids, nct_ids

    def _claim_supported_by_chunk(self, claim: str, chunk_text: str) -> float:
        """
        Score how well a chunk supports a claim using NLI.

        Args:
            claim: Claim sentence
            chunk_text: Text from retrieved chunk

        Returns:
            Support score between 0 and 1 (1 = fully supported, 0 = contradicted)
        """
        if self.model is None:
            # Fallback: simple keyword overlap heuristic
            return self._heuristic_support_score(claim, chunk_text)

        try:
            # Use NLI model: premise = chunk_text, hypothesis = claim
            # We want to know if the chunk entails the claim
            scores = self.model.predict([(chunk_text, claim)])
            # CrossEncoder for NLI typically returns scores for [contradiction, neutral, entailment]
            # Assuming the model is trained for standard NLI with 3 classes
            if isinstance(scores[0], list) and len(scores[0]) >= 3:
                # Scores are [contradiction, neutral, entailment]
                contradiction, neutral, entailment = scores[0][:3]
                # Convert to support score: entailment -> high support, contradiction -> low support
                support_score = entailment / (contradiction + neutral + entailment)
                return float(support_score)
            else:
                # Fallback if output format is unexpected
                return self._heuristic_support_score(claim, chunk_text)
        except Exception as e:
            print(f"Error in NLI scoring: {e}")
            return self._heuristic_support_score(claim, chunk_text)

    def _heuristic_support_score(self, claim: str, chunk_text: str) -> float:
        """
        Heuristic support score based on keyword overlap.

        Args:
            claim: Claim sentence
            chunk_text: Text from retrieved chunk

        Returns:
            Support score between 0 and 1
        """
        # Simple word overlap score
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        chunk_words = set(re.findall(r'\b\w+\b', chunk_text.lower()))

        if not claim_words:
            return 0.0

        overlap = claim_words.intersection(chunk_words)
        # Also check for important medical terms that should be present
        medical_terms = {'metformin', 'liver', 'hepatotoxicity', 'elderly', 'patients',
                        'ALT', 'AST', 'bilirubin', 'risk', 'side effect', 'adverse'}
        claim_medical = claim_words.intersection(medical_terms)
        chunk_medical = chunk_words.intersection(medical_terms)
        medical_overlap = claim_medical.intersection(chunk_medical)

        # Combine general overlap with medical term overlap
        if len(claim_words) > 0:
            word_overlap_score = len(overlap) / len(claim_words)
        else:
            word_overlap_score = 0.0

        if len(claim_medical) > 0:
            medical_overlap_score = len(medical_overlap) / len(claim_medical)
        else:
            medical_overlap_score = 1.0  # No medical terms in claim, don't penalize

        # Weighted combination
        return 0.7 * word_overlap_score + 0.3 * medical_overlap_score

    def score_answer(self, answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Score the attribution of claims in an answer to retrieved context chunks.

        Args:
            answer: Generated answer text
            context_chunks: List of retrieved chunk dictionaries used for generation

        Returns:
            Dictionary with claim scores, overall attribution score, and warnings
        """
        claims = self._split_into_claims(answer)
        if not claims:
            return {
                "claim_scores": [],
                "overall_score": 1.0,  # No claims to score
                "warnings": []
            }

        claim_scores = []
        warnings = []

        # Prepare chunk texts
        chunk_texts = []
        chunk_metadata = []
        for chunk in context_chunks:
            text = chunk.get("text", "")
            if text:
                chunk_texts.append(text)
                chunk_metadata.append(chunk.get("metadata", {}))

        if not chunk_texts:
            # No context chunks - all claims are unsupported
            claim_scores = [0.0] * len(claims)
            warnings.append("No context chunks available for attribution checking")
            return {
                "claim_scores": claim_scores,
                "overall_score": 0.0,
                "warnings": warnings
            }

        # Score each claim against all chunks
        for claim_idx, claim in enumerate(claims):
            # Extract citations from claim
            pmids, nct_ids = self._extract_citations_from_claim(claim)

            # If claim has no citations, it's unsupported
            if not pmids and not nct_ids:
                claim_scores.append(0.0)
                warnings.append(f"Claim '{claim[:50]}...' contains no citations")
                continue

            # Find best supporting chunk for this claim
            best_support_score = 0.0
            best_chunk_idx = -1

            for chunk_idx, chunk_text in enumerate(chunk_texts):
                support_score = self._claim_supported_by_chunk(claim, chunk_text)
                if support_score > best_support_score:
                    best_support_score = support_score
                    best_chunk_idx = chunk_idx

            claim_scores.append(best_support_score)

            # Warn if support score is low
            if best_support_score < 0.5:
                citation_info = ""
                if pmids or nct_ids:
                    citation_info = f" (cites PMID:{','.join(pmids)} NCT:{','.join(nct_ids)})"
                warnings.append(
                    f"Low attribution score ({best_support_score:.2f}) for claim: "
                    f"'{claim[:100]}...'{citation_info}"
                )

        # Calculate overall attribution score (average of claim scores)
        overall_score = np.mean(claim_scores) if claim_scores else 0.0

        return {
            "claim_scores": claim_scores,
            "overall_score": float(overall_score),
            "warnings": warnings
        }


# Convenience functions
def score_attribution(answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score attribution of answer to context chunks.

    Args:
        answer: Generated answer text
        context_chunks: List of retrieved chunk dictionaries

    Returns:
        Attribution scoring results
    """
    scorer = AttributionScorer()
    return scorer.score_answer(answer, context_chunks)


def filter_low_attribution_claims(answer: str,
                                 context_chunks: List[Dict[str, Any]],
                                 threshold: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Filter out claims with low attribution scores and return cleaned answer.

    Args:
        answer: Generated answer text
        context_chunks: List of retrieved chunk dictionaries
        threshold: Minimum attribution score to keep a claim

    Returns:
        Tuple of (filtered_answer, removed_claims_info)
    """
    scorer = AttributionScorer()
    result = scorer.score_answer(answer, context_chunks)

    claims = scorer._split_into_claims(answer)
    claim_scores = result["claim_scores"]

    # Keep claims with score >= threshold
    kept_claims = []
    removed_claims = []

    for claim, score in zip(claims, claim_scores):
        if score >= threshold:
            kept_claims.append(claim)
        else:
            removed_claims.append({"claim": claim, "score": score})

    # Reconstruct answer with kept claims
    # This is simplistic - in practice, we'd want to preserve formatting better
    if kept_claims:
        filtered_answer = ". ".join(kept_claims) + "."
    else:
        filtered_answer = "Insufficient evidence in retrieved sources."

    return filtered_answer, removed_claims


if __name__ == "__main__":
    # Test the attribution scorer
    print("AttributionScorer initialized successfully")

    # Test claim splitting
    scorer = AttributionScorer()
    test_answer = "Metformin reduces liver enzyme levels in elderly patients [PMID:12345678]. " \
                  "However, it may cause gastrointestinal side effects [PMID:87654321]. " \
                  "This unsupported claim has no citation."

    claims = scorer._split_into_claims(test_answer)
    print(f"\nTest answer: {test_answer}")
    print(f"Split into claims: {claims}")

    # Test citation extraction
    for claim in claims:
        pmids, nctids = scorer._extract_citations_from_claim(claim)
        print(f"Claim: '{claim}'")
        print(f"  PMIDs: {pmids}, NCTs: {nctids}")