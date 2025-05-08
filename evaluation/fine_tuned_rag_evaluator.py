import os
import json
import logging
import re
import tempfile
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sentence_transformers import SentenceTransformer
from implementation.agent_framework import SpatialAnalysisAgent

# Set environment variable to avoid tokenizers parallelism warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpatialRAGComponent:
    """Improved Retrieval-Augmented Generation component for spatial analysis queries with FAISS."""

    def __init__(self, data_dir: str, samples_path: Optional[str] = None):
        """
        Initialize the improved RAG component with FAISS.

        Args:
            data_dir: Directory containing the geospatial datasets
            samples_path: Path to the samples.json file with training examples
        """
        self.data_dir = data_dir
        self.samples_path = samples_path or os.path.join(
            data_dir, 'formatted_samples_combined.json')

        # Load knowledge base
        self.knowledge_base = self._initialize_knowledge_base()

        # Schema enforcement items are separated for priority retrieval
        self.schema_items = [item for item in self.knowledge_base
                             if "Schema" in item.get("title", "")]

        # Initialize sentence transformer for better embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Build FAISS index
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

        # Add schema information with stronger emphasis
        knowledge_base.extend(self._generate_schema_knowledge())

        # Add code examples from samples
        if os.path.exists(self.samples_path):
            try:
                with open(self.samples_path, 'r', encoding='utf-8-sig') as f:
                    samples = json.load(f)

                # Filter samples to ensure they use correct schema
                valid_samples = []
                for sample in samples:
                    code = sample.get("Code", "")
                    # Skip examples with incorrect schema references
                    if "'zoning'" in code or "'category'" in code or "'name'" in code or "'land_use'" in code:
                        continue
                    valid_samples.append(sample)

                knowledge_base.extend(
                    self._generate_code_knowledge(valid_samples))
                logger.info(
                    f"Loaded {len(valid_samples)} validated code examples for knowledge base")
            except Exception as e:
                logger.error(f"Error loading samples: {e}")

        return knowledge_base

    def _generate_schema_knowledge(self) -> List[Dict[str, Any]]:
        """
        Generate knowledge items for dataset schemas with stronger warnings.

        Returns:
            List of schema knowledge items
        """
        schema_knowledge = []

        # Parcels schema - enhanced with warnings
        parcels_schema = {
            "title": "Cambridge Parcels Schema",
            "content": """
            Cambridge Parcels Dataset (cambridge_parcels.geojson):
            - 'ml': Parcel ID (string) - ALWAYS use this for the final result
            - 'use_code': Land use code (string) - DO NOT use 'land_use' or 'zoning'
            - 'land_area': Size in square feet (numeric) - DO NOT use 'area' or 'size'
            - 'geometry': Spatial geometry of the parcel
            
            Detailed land use codes by category:
            
            Commercial use codes: '300' (HOTEL), '302' (INN-RESORT), '316' (WAREHOUSE), '323' (SH-CNTR/MALL), 
            '324' (SUPERMARKET), '325' (RETAIL-STORE), '326' (EATING-ESTBL), '327' (RETAIL-CONDO), 
            '330' (AUTO-SALES), '332' (AUTO-REPAIR), '334' (GAS-STATION), '340' (GEN-OFFICE), 
            '341' (BANK), '343' (OFFICE-CONDO), '345' (RETAIL-OFFIC), '346' (INV-OFFICE), 
            '353' (FRAT-ORGANIZ), '362' (THEATRE), '375' (TENNIS-CLUB), '404' (RES-&-DEV-FC), 
            '406' (HIGH-TECH), '0340' (MXD GEN-OFFICE), '0406' (MXD HIGH-TECH)
            
            Retail use codes: '323' (SH-CNTR/MALL), '324' (SUPERMARKET), '325' (RETAIL-STORE), 
            '326' (EATING-ESTBL), '327' (RETAIL-CONDO), '330' (AUTO-SALES)
            
            Office use codes: '340' (GEN-OFFICE), '341' (BANK), '343' (OFFICE-CONDO), 
            '345' (RETAIL-OFFIC), '346' (INV-OFFICE)
            
            Mixed-use codes: '0101' (MXD SNGL-FAM-RES), '0104' (MXD TWO-FAM-RES), '0105' (MXD THREE-FM-RES), 
            '0111' (MXD 4-8-UNIT-APT), '0112' (MXD >8-UNIT-APT), '0121' (MXD BOARDING-HS), '013' (MULTIUSE-RES), 
            '031' (MULTIUSE-COM), '0340' (MXD GEN-OFFICE), '0406' (MXD HIGH-TECH), '041' (MULTIUSE-IND), 
            '0942' (Higher Ed and Comm Mixed)
            
            Residential use codes: '101' (SNGL-FAM-RES), '1014' (SINGLE FAM W/AU), '102' (CONDOMINIUM), 
            '1028' (CNDO-RES-PKG), '104' (TWO-FAM-RES), '105' (THREE-FM-RES), '109' (MULTIPLE-RES), 
            '1094' (MULT-RES-2FAM), '1095' (MULT-RES-3FAM), '1098' (MULT-RES-4-8-APT), '111' (4-8-UNIT-APT), 
            '112' (>8-UNIT-APT), '113' (ASSISTED-LIV), '114' (AFFORDABLE APT), '121' (BOARDING-HSE), 
            '970' (Housing Authority), '9700' (Housing Authority), '9421' (Private College Res Units)
            
            Vacant land codes: '1062' (RES LND-IMP UND), '130' (RES-DEV-LAND), '131' (RES-PDV-LAND), 
            '132' (RES-UDV-LAND), '1322' (RES-UDV-PARK (OS) LN), '390' (COM-DEV-LAND), '391' (COM-PDV-LAND), 
            '392' (COM-UDV-LAND), '3922' (CRMCL REC LND), '440' (IND-DEV-LAND), '442' (IND-UDV-LAND), 
            '933' (Vacant Local Education), '936' (Vacant, Tax Title), '946' (Vacant (Private Ed))
            
            Industrial use codes: '400' (MANUFACTURNG), '401' (WAREHOUSE), '407' (CLEAN-MANUF), 
            '413' (RESRCH IND CND)
            """
        }

        # POI schema - enhanced with warnings
        poi_schema = {
            "title": "Cambridge POI Schema",
            "content": """
            CRITICAL SCHEMA REQUIREMENTS - POI 
            
            POI Dataset (cambridge_poi_processed.geojson):
            - 'business_type': Type of business/POI - NEVER use 'category', 'type', or 'name'
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
            CRITICAL SCHEMA REQUIREMENTS - CENSUS
            
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
            CRITICAL SCHEMA REQUIREMENTS - SPENDING
            
            Spending Dataset (cambridge_spend_processed.csv):
            - 'PLACEKEY': Join key with POI data
            - 'RAW_TOTAL_SPEND': Consumer spending amount
            
            For joining with POI data, use:
            poi_with_spend = poi.merge(spend, left_on='PLACEKEY', right_on='PLACEKEY', how='left')
            """
        }

        # Common errors to avoid
        common_errors = {
            "title": "Common Schema Errors to Avoid",
            "content": """
            CRITICAL SCHEMA ERRORS TO AVOID
            
            NEVER use these incorrect column names:
            - 'zoning' → Use 'use_code' instead
            - 'land_use' → Use 'use_code' instead
            - 'category' → Use 'business_type' instead
            - 'name' → For specific locations, use coordinates instead
            - 'area' → Use 'land_area' instead
            - 'type' → Use 'business_type' instead
            
            Harvard Square should be defined with coordinates, not searched by name:
            harvard_square = Point(-71.1189, 42.3736)
            harvard_gdf = gpd.GeoDataFrame(geometry=[harvard_square], crs=parcels.crs)
            """
        }

        # Spatial operations guidelines
        spatial_guidelines = {
            "title": "Spatial Analysis Guidelines",
            "content": """
            CRITICAL SPATIAL OPERATIONS GUIDANCE
            
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
            [parcels_schema, poi_schema, census_schema, spending_schema, common_errors, spatial_guidelines])
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

            # Validate code for schema correctness
            schema_errors = self._validate_schema(code)
            if schema_errors:
                logger.warning(
                    f"Skipping example with schema errors: {schema_errors}")
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

    def _validate_schema(self, code: str) -> List[str]:
        """
        Validate code for schema correctness.

        Args:
            code: Python code to validate

        Returns:
            List of schema errors, empty if valid
        """
        errors = []

        # Check for common schema errors
        if "'zoning'" in code or "['zoning']" in code:
            errors.append("Uses 'zoning' instead of 'use_code'")

        if "'land_use'" in code or "['land_use']" in code:
            errors.append("Uses 'land_use' instead of 'use_code'")

        if "'category'" in code or "['category']" in code:
            errors.append("Uses 'category' instead of 'business_type'")

        if ("'name'" in code or "['name']" in code) and "harvard" in code.lower():
            errors.append("Uses 'name' instead of coordinates for locations")

        if "'area'" in code or "['area']" in code:
            errors.append("Uses 'area' instead of 'land_area'")

        return errors

    def _build_search_index(self):
        """Build the FAISS search index for knowledge retrieval."""
        # Extract document texts
        texts = [
            f"{item['title']}\n{item['content']}" for item in self.knowledge_base]

        # Create document embeddings
        self.document_embeddings = self.model.encode(texts)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.document_embeddings)

        # Create FAISS index (IndexFlatIP = inner product, for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.document_embeddings.shape[1])
        self.index.add(self.document_embeddings)

        logger.info(f"Built FAISS search index with {len(texts)} documents")

    def retrieve_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge items for a given query.

        Args:
            query: The query text
            top_k: Number of top results to return

        Returns:
            List of knowledge items
        """
        # Always include all schema items first
        results = list(self.schema_items)

        # If we already have enough schema items, just return those
        if len(results) >= top_k:
            return results[:top_k]

        # Otherwise, search for additional relevant items

        # Create query embedding
        query_embedding = self.model.encode([query])

        # Normalize embedding for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        # Get results not already included from schema items
        existing_titles = {item['title'] for item in results}

        for idx in indices[0]:
            item = self.knowledge_base[idx]
            if item['title'] not in existing_titles:
                results.append(item)
                existing_titles.add(item['title'])

                # Stop if we have enough items
                if len(results) >= top_k:
                    break

        return results

    def find_similar_examples(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar code examples for a given query using FAISS.

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

        # Create query embedding
        query_embedding = self.model.encode([query])

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Create temporary FAISS index just for code examples
        example_embeddings = self.model.encode(
            [ex["query"] for ex in code_examples])
        faiss.normalize_L2(example_embeddings)

        example_index = faiss.IndexFlatIP(example_embeddings.shape[1])
        example_index.add(example_embeddings)

        # Search for similar examples
        scores, indices = example_index.search(
            query_embedding, min(top_k, len(code_examples)))

        # Collect results
        results = [code_examples[idx] for idx in indices[0]]

        return results

    def enhance_prompt(self, query: str) -> str:
        """
        Enhance a query with relevant knowledge and schema enforcement.

        Args:
            query: The original query

        Returns:
            Enhanced query with retrieved knowledge
        """
        # Always include ALL schema items first
        schema_knowledge = self.schema_items

        # Find similar code examples, but limit to avoid overwhelming
        similar_examples = self.find_similar_examples(query, top_k=2)

        # Additional knowledge items beyond schema
        additional_knowledge = self.retrieve_knowledge(query, top_k=3)

        # Filter out schema items from additional knowledge to avoid duplication
        schema_titles = {item['title'] for item in schema_knowledge}
        additional_knowledge = [
            item for item in additional_knowledge if item['title'] not in schema_titles]

        # Build enhanced prompt with stronger schema emphasis
        enhanced_prompt = f"""Query: {query}

MANDATORY SCHEMA REQUIREMENTS - MUST FOLLOW EXACTLY
- ALWAYS use 'use_code' for land use information (NEVER use 'zoning' or 'land_use')
- ALWAYS use 'business_type' for POI classification (NEVER use 'category', 'type', or 'name')
- ALWAYS use 'land_area' for parcel size (NEVER use 'area' or 'size')
- For specific locations, ALWAYS use coordinates, not name searching
- ALWAYS use EPSG:26986 projection for accurate distance measurements

SCHEMA DETAILS:
"""

        # Add schema knowledge
        for item in schema_knowledge:
            enhanced_prompt += f"\n{item['content']}\n"

        # Add additional knowledge if available
        if additional_knowledge:
            enhanced_prompt += "\nADDITIONAL KNOWLEDGE:\n"
            for item in additional_knowledge:
                enhanced_prompt += f"\n{item['content']}\n"

        # Add similar examples if available
        if similar_examples:
            enhanced_prompt += "\nSIMILAR EXAMPLES:\n"
            for i, example in enumerate(similar_examples):
                enhanced_prompt += f"\nExample {i+1}:\nQuery: {example['query']}\n\nCode:\n{example['code']}\n"

        # Add final code validation reminder
        enhanced_prompt += f"""\nNow, generate Python code to answer the original query. Before submitting your code, verify that:
1. You're using the CORRECT column names ('use_code', 'business_type', 'land_area')
2. You're NOT using incorrect column names ('zoning', 'category', 'name', 'land_use', 'area')
3. You're using coordinates for locations, not name searching

ORIGINAL QUERY: {query}"""

        return enhanced_prompt

    def post_process_code(self, code: str) -> str:
        """
        Post-process generated code to fix any remaining schema issues.

        Args:
            code: Generated Python code

        Returns:
            Corrected Python code
        """
        # Fix common schema errors
        code = code.replace("'zoning'", "'use_code'")
        code = code.replace("['zoning']", "['use_code']")
        code = code.replace("'land_use'", "'use_code'")
        code = code.replace("['land_use']", "['use_code']")
        code = code.replace("'category'", "'business_type'")
        code = code.replace("['category']", "['business_type']")
        code = code.replace("'area'", "'land_area'")
        code = code.replace("['area']", "['land_area']")

        # Fix Harvard Square name searching with coordinate definition
        if "harvard_square = poi[poi['name']" in code:
            code = code.replace(
                "harvard_square = poi[poi['name'].str.contains('Harvard Square', case=False, na=False)]",
                "# Define Harvard Square location (fixed coordinates)\nharvard_square = Point(-71.1189, 42.3736)\nharvard_gdf = gpd.GeoDataFrame(geometry=[harvard_square], crs=parcels.crs)"
            )

        return code


class FineTunedRAGEvaluator:
    """Improved evaluator combining fine-tuned model with RAG for spatial analysis queries."""

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
            data_dir, os.path.join(data_dir, 'formatted_samples_combined.json'))

        logger.info("Improved Fine-Tuned RAG evaluator initialized")

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

                # System prompt for fine-tuned model - simplify since RAG adds context
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

                # Post-process the code to fix any remaining schema issues
                if result.get("data") and result["data"].get("code"):
                    # Apply post-processing
                    original_code = result["data"]["code"]
                    corrected_code = self.rag_component.post_process_code(
                        original_code)

                    # Replace the code if corrections were made
                    if corrected_code != original_code:
                        logger.info(
                            "Post-processing corrected schema issues in the code")
                        result["data"]["code"] = corrected_code

                        # Re-execute the corrected code
                        success, output, parcel_ids = agent.execute_code(
                            corrected_code)

                        if success:
                            result["data"]["parcel_ids"] = parcel_ids
                            result["data"]["output"] = output

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
                    metrics["method"] = "fine-tuned-rag-improved"
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
                        "method": "fine-tuned-rag-improved",
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
                    "method": "fine-tuned-rag-improved",
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
