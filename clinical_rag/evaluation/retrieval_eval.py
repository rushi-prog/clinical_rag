"""
Retrieval evaluation using BioASQ dataset.
"""
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from tqdm import tqdm
from ..retrieval.hybrid_retriever import HybridRetriever
from ..config import settings
import urllib.request
import zipfile
import os


class RetrievalEvaluator:
    def __init__(self, retriever: HybridRetriever = None):
        """
        Initialize retrieval evaluator.

        Args:
            retriever: HybridRetriever instance for evaluation
        """
        self.retriever = retriever or HybridRetriever()

    def download_bioasq(self, output_dir: str = "bioasq_data") -> str:
        """
        Download BioASQ training data for task B (factoid questions).

        Args:
            output_dir: Directory to save downloaded data

        Returns:
            Path to the downloaded data file
        """
        os.makedirs(output_dir, exist_ok=True)
        bioasq_url = "http://participants-area.bioasq.org/datasets/BioASQ-trainingData/BioASQ-trainingDatasetB.zip"
        zip_path = os.path.join(output_dir, "bioasq_training_b.zip")

        if not os.path.exists(zip_path):
            print("Downloading BioASQ training data...")
            try:
                urllib.request.urlretrieve(bioasq_url, zip_path)
                print(f"Downloaded to {zip_path}")
            except Exception as e:
                print(f"Error downloading BioASQ data: {e}")
                return ""

        # Extract if not already extracted
        extract_path = os.path.join(output_dir, "extracted")
        if not os.path.exists(extract_path):
            print("Extracting BioASQ data...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"Extracted to {extract_path}")
            except Exception as e:
                print(f"Error extracting BioASQ data: {e}")
                return ""

        # Find the JSON file
        json_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        if json_files:
            return json_files[0]
        else:
            print("No JSON file found in BioASQ data")
            return ""

    def load_bioasq_questions(self, json_path: str, max_questions: int = None) -> List[Dict[str, Any]]:
        """
        Load questions from BioASQ JSON file.

        Args:
            json_path: Path to BioASQ JSON file
            max_questions: Maximum number of questions to load

        Returns:
            List of question dictionaries with 'id', 'question', and 'ideal_answer'
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            questions = data.get('questions', [])
            if max_questions:
                questions = questions[:max_questions]

            # Format questions for evaluation
            formatted_questions = []
            for q in questions:
                formatted_questions.append({
                    'id': q.get('id', ''),
                    'question': q.get('body', ''),
                    'ideal_answer': q.get('ideal_answer', []),
                    'exact_answer': q.get('exact_answer', []),
                    'snippets': q.get('snippets', [])  # These contain relevant document snippets
                })

            return formatted_questions
        except Exception as e:
            print(f"Error loading BioASQ questions: {e}")
            return []

    def evaluate_recall_at_k(self,
                            questions: List[Dict[str, Any]],
                            k_values: List[int] = [1, 5, 10],
                            save_predictions: bool = False) -> Dict[str, float]:
        """
        Evaluate Recall@k for the retrieval system.

        Args:
            questions: List of question dictionaries
            k_values: List of k values to evaluate Recall@k
            save_predictions: Whether to save predictions to file

        Returns:
            Dictionary with Recall@k scores for each k
        """
        if not self.retriever:
            print("Error: No retriever provided for evaluation")
            return {f"Recall@{k}": 0.0 for k in k_values}

        recall_scores = {f"Recall@{k}": 0.0 for k in k_values}
        total_questions = len(questions)

        if total_questions == 0:
            return recall_scores

        print(f"Evaluating Recall@k for {total_questions} questions...")

        # For each question, retrieve chunks and check if gold standard appears in top-k
        for question in tqdm(questions, desc="Evaluating questions"):
            question_text = question['question']
            # Extract relevant document IDs from ideal_answer or snippets
            # In BioASQ, ideal_answer contains relevant document snippets
            # We'll extract document IDs from snippets for evaluation
            relevant_docs = set()

            # From ideal_answer (if available)
            for answer in question.get('ideal_answer', []):
                # Extract document IDs from answer text (simplified)
                # In practice, you'd map these to your indexed documents
                pass

            # From snippets (more reliable for BioASQ)
            for snippet in question.get('snippets', []):
                doc_id = snippet.get('document', '')
                if doc_id:
                    # Extract PMID or other ID from document URL
                    # Document URLs often look like: "https://www.ncbi.nlm.nih.gov/pubmed/12345678"
                    import re
                    pmid_match = re.search(r'pubmed/(\d+)', doc_id)
                    if pmid_match:
                        relevant_docs.add(pmid_match.group(1))

            if not relevant_docs:
                # Skip questions without clear gold standard documents
                continue

            # Retrieve top-k chunks for each k value
            max_k = max(k_values)
            retrieved_chunks = self.retriever.retrieve(question_text, top_k=max_k)

            # Extract document IDs from retrieved chunks
            retrieved_doc_ids = set()
            for chunk in retrieved_chunks:
                metadata = chunk.get('metadata', {})
                pmid = metadata.get('pmid')
                if pmid:
                    retrieved_doc_ids.add(str(pmid))

            # Calculate Recall@k for each k
            for k in k_values:
                top_k_retrieved = set(list(retrieved_doc_ids)[:k])
                if relevant_docs.intersection(top_k_retrieved):
                    recall_scores[f"Recall@{k}"] += 1

        # Average scores
        for k in k_values:
            recall_scores[f"Recall@{k}"] /= total_questions

        return recall_scores

    def evaluate_mrr(self, questions: List[Dict[str, Any]]) -> float:
        """
        Evaluate Mean Reciprocal Rank (MRR) for the retrieval system.

        Args:
            questions: List of question dictionaries

        Returns:
            MRR score
        """
        if not self.retriever:
            print("Error: No retriever provided for evaluation")
            return 0.0

        total_questions = len(questions)
        if total_questions == 0:
            return 0.0

        reciprocal_ranks = []

        print(f"Evaluating MRR for {total_questions} questions...")
        for question in tqdm(questions, desc="Evaluating MRR"):
            question_text = question['question']

            # Extract relevant document IDs from snippets
            relevant_docs = set()
            for snippet in question.get('snippets', []):
                doc_id = snippet.get('document', '')
                if doc_id:
                    import re
                    pmid_match = re.search(r'pubmed/(\d+)', doc_id)
                    if pmid_match:
                        relevant_docs.add(pmid_match.group(1))

            if not relevant_docs:
                continue

            # Retrieve a reasonable number of chunks (e.g., top 100)
            retrieved_chunks = self.retriever.retrieve(question_text, top_k=100)

            # Find rank of first relevant document
            retrieved_doc_ids = []
            for chunk in retrieved_chunks:
                metadata = chunk.get('metadata', {})
                pmid = metadata.get('pmid')
                if pmid:
                    retrieved_doc_ids.append(str(pmid))

            # Find first relevant document
            rank = 0
            for i, doc_id in enumerate(retrieved_doc_ids):
                if doc_id in relevant_docs:
                    rank = i + 1  # 1-indexed rank
                    break

            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)  # No relevant document found

        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        return mrr

    def run_retrieval_evaluation(self,
                               num_questions: int = 100,
                               k_values: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """
        Run complete retrieval evaluation using BioASQ data.

        Args:
            num_questions: Number of questions to evaluate
            k_values: List of k values for Recall@k

        Returns:
            Dictionary with evaluation results
        """
        print("Starting retrieval evaluation using BioASQ data...")

        # Download BioASQ data
        bioasq_path = self.download_bioasq()
        if not bioasq_path or not os.path.exists(bioasq_path):
            print("Failed to download BioASQ data. Using mock evaluation.")
            return self._mock_evaluation(num_questions, k_values)

        # Load questions
        questions = self.load_bioasq_questions(bioasq_path, max_questions=num_questions)
        if not questions:
            print("Failed to load BioASQ questions. Using mock evaluation.")
            return self._mock_evaluation(num_questions, k_values)

        print(f"Loaded {len(questions)} questions from BioASQ")

        # Evaluate Recall@k
        recall_results = self.evaluate_recall_at_k(questions, k_values=k_values)

        # Evaluate MRR
        mrr_score = self.evaluate_mrr(questions)

        results = {
            **recall_results,
            "MRR": mrr_score,
            "num_questions": len(questions)
        }

        return results

    def _mock_evaluation(self, num_questions: int, k_values: List[int]) -> Dict[str, Any]:
        """
        Provide mock evaluation results when BioASQ data is not available.

        Args:
            num_questions: Number of questions (for reporting)
            k_values: List of k values

        Returns:
            Dictionary with mock evaluation results
        """
        print(f"Running mock evaluation for {num_questions} questions...")
        # Return reasonable mock scores
        results = {}
        for k in k_values:
            # Mock Recall@k scores (would be higher with good retrieval)
            results[f"Recall@{k}"] = 0.6 + (0.1 * (k / max(k_values)))  # Increase with k
        results["MRR"] = 0.55
        results["num_questions"] = num_questions
        results["note"] = "Mock results - replace with actual BioASQ evaluation"
        return results


# Convenience functions
def evaluate_retrieval(retriever: HybridRetriever = None,
                      num_questions: int = 100) -> Dict[str, Any]:
    """
    Convenience function to run retrieval evaluation.

    Args:
        retriever: HybridRetriever instance
        num_questions: Number of questions to evaluate

    Returns:
        Dictionary with evaluation scores
    """
    evaluator = RetrievalEvaluator(retriever)
    return evaluator.run_retrieval_evaluation(num_questions=num_questions)


if __name__ == "__main__":
    # Test the retrieval evaluator
    print("RetrievalEvaluator initialized successfully")
    print("Requires proper retriever initialization for full functionality")

    # Example usage (would need actual retriever)
    # evaluator = RetrievalEvaluator()
    # results = evaluator.run_retrieval_evaluation(num_questions=50)
    # print(f"Retrieval evaluation results: {results}")