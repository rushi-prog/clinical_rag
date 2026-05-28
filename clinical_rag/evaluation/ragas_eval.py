"""
RAGAS evaluation pipeline for faithfulness and relevancy metrics.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from ..generation.chain import create_rag_chain
from ..config import settings
import json


class RAGEvaluator:
    def __init__(self, rag_chain=None):
        """
        Initialize RAGAS evaluator.

        Args:
            rag_chain: RAG chain instance for generating answers
        """
        self.rag_chain = rag_chain

    def prepare_evaluation_dataset(self,
                                 questions: List[str],
                                 ground_truths: List[str],
                                 contexts: List[List[str]] = None) -> Dataset:
        """
        Prepare dataset for RAGAS evaluation.

        Args:
            questions: List of questions
            ground_truths: List of ground truth answers
            contexts: Optional list of context lists for each question

        Returns:
            Hugging Face Dataset for RAGAS evaluation
        """
        data = {
            "question": questions,
            "answer": [],  # Will be filled by RAG chain
            "contexts": [] if contexts is None else contexts,
            "ground_truth": ground_truths
        }

        # If RAG chain is provided, generate answers
        if self.rag_chain and contexts is None:
            print("Generating answers using RAG chain...")
            for question in questions:
                try:
                    result = self.rag_chain.invoke({"question": question})
                    # Extract answer from chain result
                    if isinstance(result, dict):
                        answer = result.get("answer", str(result))
                    else:
                        answer = str(result)
                    data["answer"].append(answer)
                except Exception as e:
                    print(f"Error generating answer for question: {question}")
                    print(f"Error: {e}")
                    data["answer"].append("Error generating answer")
        elif contexts is not None:
            # Use provided contexts
            data["contexts"] = contexts
            # Generate answers using contexts (simplified)
            for i, question in enumerate(questions):
                data["answer"].append(f"Answer based on provided context for: {question}")

        return Dataset.from_dict(data)

    def evaluate_faithfulness(self, dataset: Dataset) -> float:
        """
        Evaluate faithfulness using RAGAS.

        Args:
            dataset: Dataset with question, answer, contexts

        Returns:
            Faithfulness score
        """
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness]
            )
            return result['faithfulness']
        except Exception as e:
            print(f"Error evaluating faithfulness: {e}")
            return 0.0

    def evaluate_answer_relevancy(self, dataset: Dataset) -> float:
        """
        Evaluate answer relevancy using RAGAS.

        Args:
            dataset: Dataset with question, answer

        Returns:
            Answer relevancy score
        """
        try:
            result = evaluate(
                dataset,
                metrics=[answer_relevancy]
            )
            return result['answer_relevancy']
        except Exception as e:
            print(f"Error evaluating answer relevancy: {e}")
            return 0.0

    def evaluate_context_precision(self, dataset: Dataset) -> float:
        """
        Evaluate context precision using RAGAS.

        Args:
            dataset: Dataset with question, answer, contexts, ground_truth

        Returns:
            Context precision score
        """
        try:
            result = evaluate(
                dataset,
                metrics=[context_precision]
            )
            return result['context_precision']
        except Exception as e:
            print(f"Error evaluating context precision: {e}")
            return 0.0

    def evaluate_context_recall(self, dataset: Dataset) -> float:
        """
        Evaluate context recall using RAGAS.

        Args:
            dataset: Dataset with question, answer, contexts, ground_truth

        Returns:
            Context recall score
        """
        try:
            result = evaluate(
                dataset,
                metrics=[context_recall]
            )
            return result['context_recall']
        except Exception as e:
            print(f"Error evaluating context recall: {e}")
            return 0.0

    def run_full_evaluation(self, dataset: Dataset) -> Dict[str, float]:
        """
        Run full RAGAS evaluation suite.

        Args:
            dataset: Dataset for evaluation

        Returns:
            Dictionary with all metric scores
        """
        print("Running full RAGAS evaluation...")
        results = {}

        results['faithfulness'] = self.evaluate_faithfulness(dataset)
        results['answer_relevancy'] = self.evaluate_answer_relevancy(dataset)
        results['context_precision'] = self.evaluate_context_precision(dataset)
        results['context_recall'] = self.evaluate_context_recall(dataset)

        return results


# Convenience functions
def create_eval_dataset_from_file(filepath: str) -> Dataset:
    """
    Create evaluation dataset from JSON file.

    Expected format:
    [
        {
            "question": "What is X?",
            "ground_truth": "Y is the answer.",
            "contexts": ["Context 1", "Context 2"]
        }
    ]
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    ground_truths = [item["ground_truth"] for item in data]
    contexts = [item.get("contexts", []) for item in data]

    return Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts
    })


def evaluate_rag_system(rag_chain,
                       eval_questions: List[str],
                       ground_truths: List[str]) -> Dict[str, float]:
    """
    Convenience function to evaluate RAG system.

    Args:
        rag_chain: RAG chain instance
        eval_questions: List of evaluation questions
        ground_truths: List of ground truth answers

    Returns:
        Dictionary with evaluation scores
    """
    evaluator = RAGEvaluator(rag_chain)
    dataset = evaluator.prepare_evaluation_dataset(eval_questions, ground_truths)
    return evaluator.run_full_evaluation(dataset)


if __name__ == "__main__":
    # Test the evaluator
    print("RAGEvaluator initialized successfully")
    print("Requires RAGAS and datasets packages for full functionality")

    # Example usage (would need actual RAG chain)
    # sample_questions = [
    #     "What are the liver effects of metformin in elderly patients?",
    #     "Does atorvastatin reduce cardiovascular risk in diabetic patients?"
    # ]
    # sample_ground_truths = [
    #     "Metformin reduces ALT levels in elderly patients with type 2 diabetes.",
    #     "Atorvastatin significantly reduces cardiovascular events in diabetic patients."
    # ]
    # evaluator = RAGEvaluator()
    # dataset = evaluator.prepare_evaluation_dataset(sample_questions, sample_ground_truths)
    # results = evaluator.run_full_evaluation(dataset)
    # print(f"Evaluation results: {results}")