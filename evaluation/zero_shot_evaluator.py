import os
import json
import logging
import re
import tempfile
import time
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from implementation.agent_framework import SpatialAnalysisAgent

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ZeroShotEvaluator:
    """Evaluator for zero-shot learning approach using the SpatialAnalysisAgent."""

    def __init__(self, data_dir: str, test_samples: List[Dict], data_files: Dict[str, str], openai_api_key: str = os.getenv("OPENAI_API_KEY")):
        """
        Initialize the evaluator.

        Args:
            data_dir: Directory containing the geospatial datasets
            test_samples: List of test sample dictionaries
            data_files: Dictionary of data file paths
            openai_api_key: OpenAI API key for querying models
        """
        self.data_dir = data_dir
        self.test_samples = test_samples
        self.data_files = data_files
        self.openai_api_key = openai_api_key

        # Initialize SpatialAnalysisAgent with zero-shot system prompt
        self.zero_shot_prompt = """You are an expert in geospatial data analysis using Python. 
You translate natural language queries about site selection into Python code.
You focus on generating concise, executable Python code without examples or explanations.
Analyze the user's query, identify the spatial constraints, and write code that uses GeoPandas to find matching parcels."""

        # We'll create an agent for each evaluation to avoid state interference
        logger.info("Zero-Shot evaluator initialized")

    def evaluate(self, test_samples: List[Dict] = None) -> List[Dict]:
        """
        Evaluate the zero-shot learning approach on the provided test samples.

        Args:
            test_samples: List of test samples to evaluate (None for all)

        Returns:
            List of evaluation result dictionaries
        """
        if test_samples is None:
            # Use all test samples if none provided
            test_samples = self.test_samples

        results = []

        for sample in tqdm(test_samples, desc="Zero-shot evaluation"):
            query = sample["Query"]
            ground_truth_ids = eval(sample["Answer"]) if isinstance(
                sample["Answer"], str) else sample["Answer"]

            # Log the current query being evaluated
            logger.info(f"Evaluating zero-shot on query: {query}")

            # Create a fresh agent for each evaluation
            agent = SpatialAnalysisAgent(
                data_dir=self.data_dir,
                model_name="gpt-4o",  # Using GPT-4o for best code generation
                openai_api_key=self.openai_api_key,
                system_prompt=self.zero_shot_prompt
            )

            try:
                # Run the query through the agent
                result = agent.run_conversation(query)

                # Check if code was generated and executed successfully
                if (result.get("data") and
                    result["data"].get("code") and
                        result["data"].get("parcel_ids")):

                    code = result["data"]["code"]
                    generated_ids = result["data"]["parcel_ids"]

                    # Calculate metrics
                    metrics = self.calculate_metrics(
                        generated_ids, ground_truth_ids)
                    metrics["query"] = query
                    metrics["method"] = "zero-shot"
                    metrics["success"] = True
                    metrics["code"] = code
                    metrics["generated_ids"] = generated_ids
                    results.append(metrics)

                    # Log the results
                    logger.info(f"Zero-shot metrics for query '{query[:30]}...': "
                                f"F1={metrics['f1_score']:.3f}, "
                                f"Precision={metrics['precision']:.3f}, "
                                f"Recall={metrics['recall']:.3f}")
                else:
                    # Log the failure
                    logger.warning(f"Zero-shot failed for query: {query}")
                    results.append({
                        "query": query,
                        "method": "zero-shot",
                        "success": False,
                        "code": result["data"].get("code", "")
                    })

                # Add a small delay to avoid hitting rate limits
                time.sleep(1)

            except Exception as e:
                logger.error(
                    f"Error in zero-shot evaluation for query {query}: {e}")
                results.append({
                    "query": query,
                    "method": "zero-shot",
                    "success": False,
                    "error": str(e)
                })

        return results

    def calculate_metrics(self, generated_ids: List[str], ground_truth_ids: List[str]) -> Dict[str, float]:
        """
        Calculate evaluation metrics by comparing generated and ground truth parcel IDs.

        Args:
            generated_ids: List of parcel IDs from generated code
            ground_truth_ids: List of parcel IDs from ground truth

        Returns:
            Dictionary of metric names and values
        """
        # Convert lists to sets for easier comparison
        generated_set = set(generated_ids)
        ground_truth_set = set(ground_truth_ids)

        # Calculate metrics
        correct_ids = generated_set.intersection(ground_truth_set)

        # Handle edge cases
        if len(generated_set) == 0 and len(ground_truth_set) == 0:
            precision = 1.0
        elif len(generated_set) == 0:
            precision = 0.0
        else:
            precision = len(correct_ids) / len(generated_set)

        if len(ground_truth_set) == 0:
            recall = 1.0
        else:
            recall = len(correct_ids) / len(ground_truth_set)

        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # Result match rate
        result_match_rate = len(correct_ids) / max(len(generated_set), 1)

        # Exact match (binary accuracy)
        exact_match = 1.0 if generated_set == ground_truth_set else 0.0

        return {
            "exact_match": exact_match,
            "result_match_rate": result_match_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "generated_count": len(generated_set),
            "ground_truth_count": len(ground_truth_set),
            "correct_count": len(correct_ids)
        }
