import os
import json
import logging
import re
import tempfile
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from implementation.agent_framework import SpatialAnalysisAgent

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpatialRAGComponent:
    """Retrieval-Augmented Generation component for spatial analysis queries."""

    def __init__(self, data_dir: str, samples_path: Optional[str] = None):
        """
        Initialize the RAG component.

        Args:
            data_dir: Directory containing the geospatial datasets
            samples_path: Path to the samples.json file with training examples
        """
        self.data_dir = data_dir
        self.samples_path = samples_path or os.path.join(
            data_dir, 'spatial_samples.json')

        # Load knowledge base
        self.knowledge_base = self._initialize_knowledge_base()

        # Create vectorizer for semantic search
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Build search index
        self._build_search_index()

        logger.info(
            f"Spatial RAG component initialized with {len(self.knowledge_base)} knowledge items")

    def _initialize_knowledge_base(self) -> List[Dict[str, Any]]:
        """
        Initialize the knowledge base with schema information and code examples.

        Returns:
            List of knowledge items
        """
        knowledge_base = []

        # Add schema information
        knowledge_base.extend(self._generate_schema_knowledge())

        # Add code examples from samples
        if os.path.exists(self.samples_path):
            try:
                with open(self.samples_path, 'r', encoding='utf-8-sig') as f:
                    samples = json.load(f)
                knowledge_base.extend(self._generate_code_knowledge(samples))
                logger.info(
                    f"Loaded {len(samples)} code examples for knowledge base")
            except Exception as e:
                logger.error(f"Error loading samples: {e}")

        return knowledge_base

    def _generate_schema_knowledge(self) -> List[Dict[str, Any]]:
        """
        Generate knowledge items for dataset schemas.

        Returns:
            List of schema knowledge items
        """
        schema_knowledge = []

        # Parcels schema
        parcels_schema = {
            "title": "Cambridge Parcels Schema",
            "content": """
            Cambridge Parcels Dataset (cambridge_parcels.geojson):
            - 'ml': Parcel ID (string) - ALWAYS use this for the final result
            - 'use_code': Land use code (string) - DO NOT use 'land_use' or 'zoning'
            - 'land_area': Size in square feet (numeric) - DO NOT use 'area' or 'size'
            - 'geometry': Spatial geometry of the parcel
            
            Commercial use codes: '300', '302', '316', '323', '324', '325', '326', '327', '330', '332', '334', '340', '341', '343', '345', '346', '353', '362', '375', '404', '406', '0340', '0406'
            Retail use codes: '323', '324', '325', '326', '327', '330'
            Office use codes: '340', '341', '343', '345', '346'
            Mixed-use codes: '0101', '0104', '0105', '0111', '0112', '0121', '013', '031', '0340', '0406', '041', '0942'
            Residential use codes: '101', '1014', '102', '1028', '104', '105', '109', '1094', '1095', '1098', '111', '112', '113', '114', '121', '970', '9700', '9421'
            """
        }

        # POI schema
        poi_schema = {
            "title": "Cambridge POI Schema",
            "content": """
            POI Dataset (cambridge_poi_processed.geojson):
            - 'business_type': Type of business/POI - DO NOT use 'category', 'type', or 'name'
            - 'geometry': Spatial location
            - 'PLACEKEY': Identifier for joining with spending data
            
            For restaurant filtering, always use:
            restaurants = poi[poi['business_type'] == 'restaurant']
            
            For subway stations, use these exact coordinates:
            harvard_square = Point(-71.1189, 42.3736)
            central_square = Point(-71.1031, 42.3656)
            kendall_mit = Point(-71.0865, 42.3625)
            porter_square = Point(-71.1226, 42.3782)
            alewife = Point(-71.1429, 42.3954)
            """
        }

        # Census schema
        census_schema = {
            "title": "Cambridge Census Schema",
            "content": """
            Census Dataset (cambridge_census_cambridge_pct.geojson):
            - 'pct_adv_deg': Percentage with advanced degrees
            - 'pct_18_64': Percentage aged 18-64
            - 'median_income': Median income
            
            For educational attainment filtering, use:
            high_edu_census = census[census['pct_adv_deg'] > threshold]
            """
        }

        # Spending schema
        spending_schema = {
            "title": "Cambridge Spending Schema",
            "content": """
            Spending Dataset (cambridge_spend_processed.csv):
            - 'PLACEKEY': Join key with POI data
            - 'RAW_TOTAL_SPEND': Consumer spending amount
            
            For joining with POI data, use:
            poi_with_spend = poi.merge(spend, left_on='PLACEKEY', right_on='PLACEKEY', how='left')
            """
        }

        # Spatial operations guidelines
        spatial_guidelines = {
            "title": "Spatial Analysis Guidelines",
            "content": """
            Always project data to EPSG:26986 for accurate distance calculations:
            parcels_proj = parcels.to_crs(epsg=26986)
            poi_proj = poi.to_crs(epsg=26986)
            
            For buffering, always use:
            buffered_geom = geom.buffer(distance)
            
            For spatial intersection:
            within_buffer = gdf[gdf.geometry.intersects(buffer)]
            
            For distance calculations:
            distances = gdf.geometry.distance(point)
            """
        }

        schema_knowledge.extend(
            [parcels_schema, poi_schema, census_schema, spending_schema, spatial_guidelines])
        return schema_knowledge

    def _generate_code_knowledge(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate knowledge items from code examples in samples.

        Args:
            samples: List of sample dictionaries with Query, Code, Answer

        Returns:
            List of code knowledge items
        """
        code_knowledge = []

        for sample in samples:
            query = sample.get("Query", "")
            code = sample.get("Code", "")

            # Skip empty examples
            if not query or not code:
                continue

            # Create knowledge item
            item = {
                "title": f"Code Example: {query[:50]}{'...' if len(query) > 50 else ''}",
                "content": f"Query: {query}\n\nCode:\n{code}",
                "query": query,
                "code": code
            }

            code_knowledge.append(item)

        return code_knowledge

    def _build_search_index(self):
        """Build the search index for knowledge retrieval."""
        # Extract document texts
        texts = [
            f"{item['title']}\n{item['content']}" for item in self.knowledge_base]

        # Fit vectorizer
        self.document_vectors = self.vectorizer.fit_transform(texts)

        logger.info(f"Built search index with {len(texts)} documents")

    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge items for a given query.

        Args:
            query: The query text
            top_k: Number of top results to return

        Returns:
            List of knowledge items
        """
        # Transform query
        query_vector = self.vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(
            query_vector, self.document_vectors).flatten()

        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Return top-k items
        results = [self.knowledge_base[i] for i in top_indices]

        return results

    def find_similar_examples(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Find similar code examples for a given query.

        Args:
            query: The query text
            top_k: Number of top examples to return

        Returns:
            List of similar code examples
        """
        # Filter knowledge base for code examples only
        code_examples = [
            item for item in self.knowledge_base if "code" in item]

        if not code_examples:
            return []

        # Extract texts from code examples
        texts = [item["query"] for item in code_examples]

        # Create a temporary vectorizer for code examples
        temp_vectorizer = TfidfVectorizer(stop_words='english')
        example_vectors = temp_vectorizer.fit_transform(texts)
        query_vector = temp_vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(
            query_vector, example_vectors).flatten()

        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Return top-k examples
        results = [code_examples[i] for i in top_indices]

        return results

    def enhance_prompt(self, query: str) -> str:
        """
        Enhance a query with relevant knowledge.

        Args:
            query: The original query

        Returns:
            Enhanced query with retrieved knowledge
        """
        # Retrieve schema knowledge
        schema_knowledge = self.retrieve_knowledge(query, top_k=2)

        # Find similar code examples
        similar_examples = self.find_similar_examples(query, top_k=2)

        # Build enhanced prompt
        enhanced_prompt = f"""Query: {query}

IMPORTANT SCHEMA INFORMATION:
"""

        # Add schema knowledge
        for item in schema_knowledge:
            enhanced_prompt += f"\n{item['content']}\n"

        # Add similar examples if available
        if similar_examples:
            enhanced_prompt += "\nSIMILAR EXAMPLES:\n"
            for i, example in enumerate(similar_examples):
                enhanced_prompt += f"\nExample {i+1}:\nQuery: {example['query']}\n\nCode:\n{example['code']}\n"

        enhanced_prompt += f"\nNow, generate Python code to answer the original query:\n{query}"

        return enhanced_prompt


class FineTunedRAGEvaluator:
    """Evaluator combining fine-tuned model with RAG for spatial analysis queries."""

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

        # Initialize RAG component
        self.rag_component = SpatialRAGComponent(
            data_dir, os.path.join(data_dir, 'spatial_samples.json'))

        logger.info("Fine-Tuned RAG evaluator initialized")

    def evaluate(self, test_samples: List[Dict], model_name: str) -> List[Dict]:
        """
        Evaluate the fine-tuned model with RAG on the provided test samples.

        Args:
            test_samples: List of test samples to evaluate
            model_name: Name of the fine-tuned model to use

        Returns:
            List of evaluation result dictionaries
        """
        if not model_name:
            raise ValueError("A fine-tuned model name must be provided")

        results = []

        for sample in test_samples:
            query = sample["Query"]
            ground_truth_ids = eval(sample["Answer"]) if isinstance(
                sample["Answer"], str) else sample["Answer"]

            # Log the current query being evaluated
            logger.info(
                f"Evaluating fine-tuned model with RAG {model_name} on query: {query}")

            try:
                # Enhance query with RAG
                enhanced_query = self.rag_component.enhance_prompt(query)

                # System prompt for fine-tuned model - simple since RAG adds context
                fine_tuned_prompt = """You are a commercial site selection assistant that creates Python code to analyze geospatial data in Cambridge, MA.
Generate executable GeoPandas code to find parcels matching the criteria."""

                # Create a fresh agent for this evaluation with the fine-tuned model
                agent = SpatialAnalysisAgent(
                    data_dir=self.data_dir,
                    model_name=model_name,
                    openai_api_key=self.openai_api_key,
                    system_prompt=fine_tuned_prompt
                )

                # Run the enhanced query through the agent
                result = agent.run_conversation(enhanced_query)

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
                    metrics["method"] = "fine-tuned-rag"
                    metrics["model"] = model_name
                    metrics["success"] = True
                    metrics["code"] = code
                    metrics["generated_ids"] = generated_ids
                    results.append(metrics)

                    # Log the results
                    logger.info(f"Fine-tuned RAG metrics for query '{query[:30]}...': "
                                f"F1={metrics['f1_score']:.3f}, "
                                f"Precision={metrics['precision']:.3f}, "
                                f"Recall={metrics['recall']:.3f}")
                else:
                    # Log the failure
                    logger.warning(
                        f"Fine-tuned RAG code execution failed for query: {query}")
                    results.append({
                        "query": query,
                        "method": "fine-tuned-rag",
                        "model": model_name,
                        "success": False,
                        "code": result["data"].get("code", "")
                    })

                # Add a small delay to avoid hitting rate limits
                time.sleep(1)

            except Exception as e:
                logger.error(
                    f"Error in fine-tuned RAG evaluation for query {query}: {e}")
                results.append({
                    "query": query,
                    "method": "fine-tuned-rag",
                    "model": model_name,
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
