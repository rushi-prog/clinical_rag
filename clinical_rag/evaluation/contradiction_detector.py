"""
Contradiction detector for identifying conflicting claims in retrieved chunks.
"""
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import CrossEncoder
import torch
import itertools
from tqdm import tqdm
from ..config import settings
import re


class ContradictionDetector:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        """
        Initialize contradiction detector with NLI model.

        Args:
            model_name: Name of the cross-encoder model for NLI
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading contradiction detection model {self.model_name} on {self.device}")
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as e:
            print(f"Warning: Could not load NLI model {model_name}: {e}")
            print("Contradiction detection will be disabled")
            self.model = None

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.

        Args:
            text: Input text

        Returns:
            List of claim sentences
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter for sentences that look like factual claims
            # (contain medical terms, numbers, or specific entities)
            if len(sentence) > 15 and any(term in sentence.lower() for term in
                                          ['metformin', 'drug', 'patient', 'trial', 'study',
                                           'risk', 'effect', 'significant', 'percentage', '%',
                                           'mg', 'dose', 'mg/dl', 'mmhg']):
                claims.append(sentence)
        return claims

    def _normalize_drug_name_in_text(self, text: str) -> str:
        """
        Normalize drug names in text to improve matching.

        Args:
            text: Input text

        Returns:
            Text with normalized drug names
        """
        # This would integrate with the drug normalizer
        # For now, return as-is
        return text

    def _extract_disease_outcome_pairs(self, claim: str) -> List[Tuple[str, str]]:
        """
        Extract (disease, outcome) pairs from a claim for contradiction detection.

        Args:
            claim: Claim sentence

        Returns:
            List of (disease, outcome) tuples
        """
        # Simplified extraction - in practice, use NER or more sophisticated IE
        pairs = []

        # Look for patterns like "drug X increases risk of Y in Z"
        # or "drug X reduces Y in patients with Z"
        lower_claim = claim.lower()

        # Common disease conditions
        diseases = ['diabetes', 'hypertension', 'hyperlipidemia', 'obesity',
                   'renal impairment', 'hepatic impairment', 'cardiovascular disease']

        # Common outcomes
        outcomes = ['risk', 'incidence', 'mortality', 'hospitalization',
                   'liver toxicity', 'kidney damage', 'bleeding', 'fracture',
                   'cancer', 'infection', 'hospitalization', 'disability']

        # Very simplified - just look for co-occurrence
        for disease in diseases:
            if disease in lower_claim:
                for outcome in outcomes:
                    if outcome in lower_claim:
                        pairs.append((disease, outcome))

        return pairs

    def detect_contradictions(self,
                            chunks: List[Dict[str, Any]],
                            threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Detect contradictory claims among retrieved chunks.

        Args:
            chunks: List of retrieved chunk dictionaries
            threshold: Contradiction score threshold (0-1, higher = more contradictory)

        Returns:
            List of contradiction dictionaries with details
        """
        if self.model is None:
            print("Warning: Contradiction detection model not loaded")
            return []

        if len(chunks) < 2:
            return []  # Need at least 2 chunks to detect contradictions

        # Extract claims from each chunk
        chunk_claims = []
        for chunk in chunks:
            text = chunk.get("text", "")
            claims = self._extract_claims(text)
            chunk_claims.append(claims)

        # Flatten claims with chunk references
        all_claims = []
        for chunk_idx, claims in enumerate(chunk_claims):
            for claim_idx, claim in enumerate(claims):
                all_claims.append({
                    "text": claim,
                    "chunk_index": chunk_idx,
                    "claim_index": claim_idx,
                    "metadata": chunks[chunk_idx].get("metadata", {})
                })

        if len(all_claims) < 2:
            return []

        # Compare pairs of claims for contradictions
        contradictions = []

        print(f"Checking {len(all_claims)} claims for contradictions...")
        for i, j in tqdm(list(itertools.combinations(range(len(all_claims)), 2)),
                        desc="Checking claim pairs"):
            claim1 = all_claims[i]["text"]
            claim2 = all_claims[j]["text"]

            # Skip if claims are too similar (likely duplicates)
            similarity = self._jaccard_similarity(claim1, claim2)
            if similarity > 0.8:
                continue

            # Check if claims are about the same topic (disease-outcome pairs)
            pairs1 = set(self._extract_disease_outcome_pairs(claim1))
            pairs2 = set(self._extract_disease_outcome_pairs(claim2))

            # If they share disease-outcome pairs, check for contradiction
            shared_pairs = pairs1.intersection(pairs2)
            if shared_pairs:
                # Use NLI model to check contradiction
                # Premise = claim1, Hypothesis = claim2
                # We're checking if claim2 contradicts claim1
                try:
                    scores = self.model.predict([(claim1, claim2)])
                    # Assuming NLI model returns [contradiction, neutral, entailment] scores
                    if isinstance(scores[0], list) and len(scores[0]) >= 3:
                        contradiction_score = float(scores[0][0])  # contradiction score
                        neutral_score = float(scores[0][1])
                        entailment_score = float(scores[0][2])

                        if contradiction_score >= threshold:
                            contradictions.append({
                                "claim1": claim1,
                                "claim2": claim2,
                                "chunk1_index": all_claims[i]["chunk_index"],
                                "chunk2_index": all_claims[j]["chunk_index"],
                                "contradiction_score": contradiction_score,
                                "neutral_score": neutral_score,
                                "entailment_score": entailment_score,
                                "shared_topics": list(shared_pairs),
                                "metadata1": all_claims[i]["metadata"],
                                "metadata2": all_claims[j]["metadata"]
                            })
                except Exception as e:
                    print(f"Error checking contradiction between claims: {e}")
                    continue

        return contradictions

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity score (0-1)
        """
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def add_contradiction_note_to_prompt(self,
                                        contradictions: List[Dict[str, Any]]) -> str:
        """
        generate a note to add to the system prompt when contradictions are detected.

        Args:
            contradictions: List of contradiction dictionaries

        Returns:
            Note string to add to system prompt
        """
        if not contradictions:
            return ""

        # Group contradictions by topic
        topic_contradictions = {}
        for cont in contradictions:
            for topic in cont["shared_topics"]:
                if topic not in topic_contradictions:
                    topic_contradictions[topic] = []
                topic_contradictions[topic].append(cont)

        # Build note
        note_parts = ["Note: Retrieved sources contain conflicting evidence on the following topics:"]
        for topic, cont_list in topic_contradictions.items():
            note_parts.append(f"- {topic}: {len(cont_list)} conflicting claim(s)")

        note_parts.append("Surface both views explicitly in your answer.")
        return "\n".join(note_parts)


# Convenience functions
def detect_contradictions(chunks: List[Dict[str, Any]],
                         threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Detect contradictions in retrieved chunks.

    Args:
        chunks: List of retrieved chunk dictionaries
        threshold: Contradiction score threshold

    Returns:
        List of contradiction dictionaries
    """
    detector = ContradictionDetector()
    return detector.detect_contradictions(chunks, threshold=threshold)


if __name__ == "__main__":
    # Test the contradiction detector
    print("ContradictionDetector initialized successfully")

    # Test claim extraction
    detector = ContradictionDetector()
    test_text = "Metformin reduces liver enzyme levels in elderly patients with type 2 diabetes. " \
                "However, some studies show metformin increases risk of lactic acidosis in renal impairment. " \
                "Atorvastatin significantly reduces cardiovascular events in diabetic patients."

    claims = detector._extract_claims(test_text)
    print(f"\nTest text: {test_text}")
    print(f"Extracted claims: {claims}")

    # Test disease-outcome extraction
    for claim in claims:
        pairs = detector._extract_disease_outcome_pairs(claim)
        print(f"Claim: '{claim}'")
        print(f"  Disease-outcome pairs: {pairs}")