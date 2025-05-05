import os
import json
import logging
import re
import tempfile
import random
import time
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from implementation.agent_framework import SpatialAnalysisAgent

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FewShotEvaluator:
    """Improved evaluator for few-shot learning with basic schema information."""

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

        # Initialize with improved but minimal schema information
        self.schema_prompt = """You are an expert in geospatial data analysis using Python. You translate natural language queries about site selection into Python code. You focus on generating concise, executable Python code without examples or explanations. Analyze the user's query, identify the spatial constraints, and write code that uses GeoPandas to find matching parcels.
ESSENTIAL DATASET SCHEMA:
1. Parcels ('cambridge_parcels.geojson'):
   - 'ml': Parcel ID (string)
   - 'use_code': Land use code (string) for commercial/retail/residential classification 
   - 'land_area': Size in square feet (numeric)
   - 'geometry': Spatial geometry of the parcel
   NOTE: NO 'zoning' column. Use 'use_code' to filter

2. POI ('cambridge_poi_processed.geojson'):
   - 'business_type': Type of business/POI (NOT 'category', 'type', or 'name')
   - 'geometry': Spatial location
   - 'PLACEKEY': Identifier for joining with spending data
   NOTE: NOT 'category', 'type', or 'name', use 'business_type' instead

3. Census ('cambridge_census_cambridge_pct.geojson'):
   - 'pct_adv_deg': % with advanced degrees
   - 'pct_18_64': % aged 18-64
   - 'median_income': Median income

4. Spending ('cambridge_spend_processed.csv'):
   - 'PLACEKEY': Join key with POI data
   - 'RAW_TOTAL_SPEND': Consumer spending amount

Use ONLY these documented fields in your code. Project to EPSG:26986 for accurate distance calculations.
"""

        logger.info(
            "Improved Few-Shot evaluator initialized with basic schema information")

    def create_few_shot_prompt(self, examples: List[Dict]) -> str:
        """Create a system prompt with few-shot examples and schema information.

        Args:
            examples: List of example dictionaries with Query, Code, Answer fields

        Returns:
            Few-shot system prompt as a string
        """
        prompt = self.schema_prompt + """Below are some examples of site selection queries and the corresponding Python code that finds matching parcels:
        """
        # Add examples to the prompt
        for i, example in enumerate(examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Query: {example['Query']}\n\n"
            prompt += f"Python Code:\n{example['Code']}\n\n"

        prompt += """When given a new query, analyze it carefully to identify the spatial constraints and requirements.
Generate Python code similar to the examples that will correctly find the parcels matching the criteria.
Make sure your code handles all the constraints mentioned in the query and follows the patterns shown in the examples.
Always print the final list of parcel IDs at the end of your code."""

        return prompt

    def evaluate(self, test_samples: List[Dict] = None, num_examples: int = 3) -> List[Dict]:
        """
        Evaluate the few-shot learning approach on the provided test samples.

        Args:
            test_samples: List of test samples to evaluate (None for all)
            num_examples: Number of examples to include in the prompt

        Returns:
            List of evaluation result dictionaries
        """
        if test_samples is None:
            # Use all test samples if none provided
            test_samples = self.test_samples

        results = []

        for sample in tqdm(test_samples, desc=f"Improved few-shot ({num_examples} examples) evaluation"):
            query = sample["Query"]
            ground_truth_ids = eval(sample["Answer"]) if isinstance(
                sample["Answer"], str) else sample["Answer"]

            # Log the current query being evaluated
            logger.info(
                f"Evaluating improved few-shot ({num_examples} examples) on query: {query}")

            try:
                # Select random examples from the test set (excluding current query)
                examples = []
                current_samples = [
                    s for s in self.test_samples if s["Query"].lower() != query.lower()]
                if len(current_samples) >= num_examples:
                    examples = random.sample(current_samples, num_examples)
                else:
                    examples = current_samples

                # Create few-shot system prompt with schema info
                few_shot_prompt = self.create_few_shot_prompt(examples)

                # Create a fresh agent for this evaluation
                agent = SpatialAnalysisAgent(
                    data_dir=self.data_dir,
                    model_name="gpt-4o",  # Using GPT-4o for best code generation
                    openai_api_key=self.openai_api_key,
                    system_prompt=few_shot_prompt
                )

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
                    metrics["method"] = f"improved-few-shot-{num_examples}"
                    metrics["success"] = True
                    metrics["code"] = code
                    metrics["generated_ids"] = generated_ids
                    results.append(metrics)

                    # Log the results
                    logger.info(f"Improved few-shot metrics for query '{query[:30]}...': "
                                f"F1={metrics['f1_score']:.3f}, "
                                f"Precision={metrics['precision']:.3f}, "
                                f"Recall={metrics['recall']:.3f}")
                else:
                    # Log the failure
                    logger.warning(
                        f"Improved few-shot code execution failed for query: {query}")
                    results.append({
                        "query": query,
                        "method": f"improved-few-shot-{num_examples}",
                        "success": False,
                        "code": result["data"].get("code", "")
                    })

                # Add a small delay to avoid hitting rate limits
                time.sleep(1)

            except Exception as e:
                logger.error(
                    f"Error in improved few-shot evaluation for query {query}: {e}")
                results.append({
                    "query": query,
                    "method": f"improved-few-shot-{num_examples}",
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
