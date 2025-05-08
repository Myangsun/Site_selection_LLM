import os
import json
import logging
import random
import time
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm
from implementation.agent_framework import SpatialAnalysisAgent

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataConstrainedSampleGenerator:
    """Generate samples constrained to available datasets with improved column validation."""

    def __init__(self, data_dir: str, fine_tuned_model: str, openai_api_key: Optional[str] = None):
        """
        Initialize the sample generator with dataset constraints.

        Args:
            data_dir: Directory containing the geospatial datasets
            fine_tuned_model: Fine-tuned model to use for generation
            openai_api_key: OpenAI API key for API access
        """
        self.data_dir = data_dir
        self.fine_tuned_model = fine_tuned_model
        self.openai_api_key = openai_api_key or os.environ.get(
            "OPENAI_API_KEY")

        # Load existing samples
        self.existing_samples = self._load_existing_samples()

        # Set up data file paths
        self.data_files = {
            'parcels': os.path.join(data_dir, 'cambridge_parcels.geojson'),
            'poi': os.path.join(data_dir, 'cambridge_poi_processed.geojson'),
            'census': os.path.join(data_dir, 'cambridge_census_cambridge_pct.geojson'),
            'spend': os.path.join(data_dir, 'cambridge_spend_processed.csv')
        }

        # Known dataset schema to constrain queries
        self.dataset_schema = {
            'parcels': {
                'columns': ['ml', 'use_code', 'land_area', 'geometry'],
                'filters': ['commercial', 'retail', 'office', 'mixed-use', 'residential', 'vacant']
            },
            'poi': {
                # POI doesn't have 'ml' column
                'columns': ['business_type', 'geometry', 'PLACEKEY'],
                'filters': ['restaurant', 'subway']
            },
            'census': {
                'columns': ['pct_adv_deg', 'pct_18_64'],
                'filters': ['education', 'demographics']
            },
            'spend': {
                # Correct column name for spending
                'columns': ['PLACEKEY', 'RAW_TOTAL_SPEND'],
                'filters': ['consumer spending']
            }
        }

        # Mapping of incorrect column names to correct ones
        self.column_corrections = {
            # POI dataset corrections
            "'ml'": "'business_type'",  # POI doesn't have ml column
            "'category'": "'business_type'",
            "'type'": "'business_type'",
            "'name'": "'business_type'",
            "'poi_type'": "'business_type'",

            # Parcels dataset corrections
            "'land_use'": "'use_code'",
            "'area'": "'land_area'",
            "'zoning'": "'use_code'",

            # Spending dataset corrections
            "'estimated_spend'": "'RAW_TOTAL_SPEND'",
            "'spending'": "'RAW_TOTAL_SPEND'",
            "'consumer_spending'": "'RAW_TOTAL_SPEND'"
        }

        logger.info(
            f"Dataset-constrained sample generator initialized with {len(self.existing_samples)} existing samples")

    def _load_existing_samples(self) -> List[Dict[str, Any]]:
        """
        Load existing samples from samples.json.

        Returns:
            List of sample dictionaries
        """
        samples_path = os.path.join(self.data_dir, 'spatial_samples.json')
        if os.path.exists(samples_path):
            try:
                with open(samples_path, 'r', encoding='utf-8-sig') as f:
                    samples = json.load(f)
                logger.info(f"Loaded {len(samples)} existing samples")
                return samples
            except Exception as e:
                logger.error(f"Error loading existing samples: {e}")

        logger.warning("No existing samples found")
        return []

    def generate_structured_queries(self, category: str, subcategory: str, count: int) -> List[str]:
        """
        Generate structured queries constrained to available dataset fields.

        Args:
            category: Main constraint category
            subcategory: Subcategory within the main category
            count: Number of queries to generate

        Returns:
            List of generated query strings
        """
        # Create detailed prompt for the specified category and subcategory
        prompt = self._create_constrained_category_prompt(
            category, subcategory, count)

        # Create a fresh agent for query generation
        agent = SpatialAnalysisAgent(
            data_dir=self.data_dir,
            model_name=self.fine_tuned_model,
            openai_api_key=self.openai_api_key
        )

        # Generate variations
        result = agent.process_message(prompt)

        # Extract JSON array from response
        message = result.get("message", "")
        try:
            # Find JSON array in the response
            json_start = message.find("[")
            json_end = message.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = message[json_start:json_end]
                queries = json.loads(json_str)

                # Filter out queries with unavailable fields
                filtered_queries = self._filter_queries_by_available_data(
                    queries)

                # Further filter for similarity to existing queries
                existing_queries = [sample["Query"]
                                    for sample in self.existing_samples]
                filtered_queries = [
                    q for q in filtered_queries
                    if not any(self._similarity(q, eq) > 0.8 for eq in existing_queries)
                ]

                logger.info(
                    f"Generated {len(filtered_queries)} valid {category}/{subcategory} queries")
                return filtered_queries[:count]  # Limit to requested count
            else:
                logger.error("Could not find JSON array in response")
                return []
        except Exception as e:
            logger.error(f"Error extracting queries: {e}")
            return []

    def _filter_queries_by_available_data(self, queries: List[str]) -> List[str]:
        """
        Filter queries to only include those that reference available dataset fields.

        Args:
            queries: List of queries to filter

        Returns:
            Filtered list of queries
        """
        # Disallowed terms - information not in datasets
        disallowed_terms = [
            "parking space", "parking spaces", "parking lot",
            "floor", "stories", "building height",
            "owner", "ownership", "tenant", "lease",
            "permit", "permits", "zoning variance",
            "median income", "median household income"  # Added to disallowed terms
        ]

        # Filter out queries that mention disallowed terms
        filtered = []
        for query in queries:
            query_lower = query.lower()
            if not any(term in query_lower for term in disallowed_terms):
                filtered.append(query)

        return filtered

    def _create_constrained_category_prompt(self, category: str, subcategory: str, count: int) -> str:
        """
        Create a prompt for generating queries, with dataset constraints emphasized.

        Args:
            category: Main constraint category
            subcategory: Subcategory within the main category
            count: Number of queries to generate

        Returns:
            Detailed prompt for generation
        """
        # Get some example queries from existing samples as inspiration
        existing_queries = [sample["Query"]
                            for sample in self.existing_samples]
        example_queries = random.sample(
            existing_queries, min(5, len(existing_queries))) if existing_queries else []

        # Basic prompt structure with dataset constraints
        base_prompt = f"""You are an expert in geospatial analysis and site selection for commercial real estate.

Here are some examples of site selection queries:
{json.dumps(example_queries, indent=2)}

IMPORTANT: You must ONLY create queries that use data from these Cambridge, MA datasets:
1. Parcels dataset: Fields include 'ml' (parcel ID), 'use_code' (for land use classification), 'land_area' (size in sq ft)
2. POI dataset: Fields include 'business_type' (type of business/POI) - NOTE: POI does NOT have an 'ml' column
3. Census dataset: Fields include 'pct_adv_deg' (% with advanced degrees), 'pct_18_64' (% aged 18-64)
4. Spending dataset: Contains 'RAW_TOTAL_SPEND' (consumer spending amount) - NOT 'estimated_spend'

DO NOT create queries about parking spaces, building height, floor count, ownership, median income, or other data NOT in these datasets.

I need you to create {count} new queries specifically for the category: "{category}" and subcategory: "{subcategory}"."""

        # Add category-specific details
        full_prompt = base_prompt + self._get_category_specific_prompt(category, subcategory) + """
Remember:
- ONLY reference data fields available in the datasets mentioned above
- NO queries about parking, building height, floors, ownership, median income, or other unavailable data
- Focus on valid spatial constraints and the available fields

Return ONLY a JSON array of query strings, with no explanations or additional text."""

        return full_prompt

    def _get_category_specific_prompt(self, category: str, subcategory: str) -> str:
        """Get category-specific prompt section."""
        if category == "Simple_Constraints":
            if subcategory == "Single Hard Constraint":
                return """
The queries should have exactly ONE hard constraint like:
- Size requirements (e.g., "larger than X square feet")
- Specific use code (e.g., "zoned for retail")
- Exact location criteria (e.g., "within X meters of [location]")

Each query should focus on finding parcels with ONE clear, non-negotiable requirement.
"""
            elif subcategory == "Single Soft Constraint":
                return """
The queries should have exactly ONE soft constraint like:
- Prioritizing certain areas (e.g., "prioritizing areas with higher foot traffic")
- Preference criteria (e.g., "preferably near residential neighborhoods")
- Optimization goals (e.g., "with the highest consumer spending")

Each query should use language indicating preference rather than requirement (e.g., "prioritizing," "preferably," "ideally").
"""
            elif subcategory == "Two Hard Constraints":
                return """
The queries should have exactly TWO hard constraints combined with "and" like:
- "larger than X square feet AND within Y meters of [location]"
- "zoned for retail AND with at least Z parking spaces"

Each query should have two clear, non-negotiable requirements that both must be satisfied.
"""
            elif subcategory == "One Hard + One Soft Constraint":
                return """
The queries should combine ONE hard constraint and ONE soft constraint like:
- "larger than X square feet, preferably in areas with high foot traffic"
- "within Y meters of subway stations, prioritizing areas with higher educational attainment"

Each query should have one clear requirement and one preference criterion.
"""
        elif category == "Complex_Constraints":
            if subcategory == "Multiple Hard Constraints":
                return """
The queries should have THREE OR MORE hard constraints like:
- "larger than X square feet, within Y meters of subway stations, and with high educational attainment"
- "zoned for retail, not within 500m of competitors, and in census tracts with high percentage of working-age residents"

Each query should have at least three non-negotiable requirements that all must be satisfied.
"""
            elif subcategory == "Multiple Mixed Constraints":
                return """
The queries should have at least TWO hard constraints AND TWO soft constraints like:
- "larger than X square feet and within Y meters of subway stations, preferably in areas with high consumer spending and diverse land uses"
- "zoned for retail and with active business licenses, ideally in areas with high educational attainment and near residential neighborhoods"

Each query should clearly distinguish between requirements and preferences.
"""
            elif subcategory == "Logical Combinations":
                return """
The queries should use complex logical combinations (AND, OR, NOT) like:
- "either office space larger than X sq ft OR retail space within Y meters of residential areas"
- "commercial parcels NOT within Z meters of industrial zones AND either near transit OR with high consumer spending"

Each query should use logical operators to create complex selection criteria.
"""
            elif subcategory == "Conditional Constraints":
                return """
The queries should use conditional (if-then) logic like:
- "parcels that, if larger than X sq ft, must be within Y meters of transit, or if smaller, must be within Z meters of residential areas"
- "mixed-use parcels that, if north of [location], must have retail use code, otherwise must have at least X sq ft of commercial space"

Each query should specify different criteria depending on some property of the parcel.
"""
        elif category == "Spatial_Constraints":
            if subcategory == "Simple Buffer Queries":
                return """
The queries should use simple buffer/distance criteria like:
- "within X meters of [location/feature]"
- "at least Y meters away from [location/feature]"
- "between X and Y meters from [location/feature]"

Each query should specify a clear spatial relationship based on distance.
"""
            elif subcategory == "Nested Spatial Relationships":
                return """
The queries should use nested or complex spatial relationships like:
- "within X meters of [feature A] AND outside Y meters of [feature B]"
- "parcels that intersect with buffer zones of [feature A] but are at least Z meters from [feature B]"
- "within X meters of [feature A] OR within Y meters of [feature B], but not within Z meters of [feature C]"

Each query should combine multiple spatial relationships in a nested or complex manner.
"""
        elif category == "Business_Environment_Constraints":
            if subcategory == "Competitor Density":
                return """
The queries should focus on competitor density/proximity like:
- "with no more than X competing businesses within Y meters"
- "at least Z meters away from nearest competitor of same type"
- "in areas with the lowest density of competing [business type]"

Each query should specify criteria related to the presence or absence of competing businesses.
"""
            elif subcategory == "Land Use Mix":
                return """
The queries should focus on the mix of surrounding land uses like:
- "in areas with diverse mix of residential and commercial uses within X meters"
- "surrounded by at least Y different land use types within Z meters"
- "in predominantly [land use type] areas but with some [other land use type] nearby"

Each query should specify criteria related to the diversity or composition of surrounding land uses.
"""
            elif subcategory == "Consumer Spending Patterns":
                return """
The queries should focus on consumer spending patterns like:
- "in areas with highest average consumer spending"
- "where RAW_TOTAL_SPEND is above city average"
- "top Y parcels ranked by total RAW_TOTAL_SPEND within Z meters"

Each query should specify criteria related to consumer spending levels or patterns.
"""
        elif category == "Demographic_Constraints":
            if subcategory == "Income Levels":
                # Modified to avoid using median income
                return """
The queries should focus on demographic characteristics like:
- "in census tracts with high educational attainment"
- "in areas where at least Y% of residents are aged 18-64"
- "in neighborhoods with highest proportion of advanced degrees"

Each query should specify criteria related to demographic characteristics of the surrounding population.
"""
            elif subcategory == "Target Demographic Match":
                return """
The queries should focus on specific demographic targets like:
- "in areas with at least X% of residents aged 18-64"
- "where percentage of residents with advanced degrees is above Y%"
- "in neighborhoods with highest concentration of working-age adults"

Each query should specify criteria related to the demographic composition of the surrounding population.
"""
        elif category == "Logical_Structure_Constraints":
            if subcategory == "AND Combinations":
                return """
The queries should use explicit AND logical combinations like:
- "parcels that are [criterion A] AND [criterion B] AND [criterion C]"
- "properties meeting ALL of the following criteria: [list criteria]"

Each query should require that multiple criteria be satisfied simultaneously.
"""
            elif subcategory == "OR Combinations":
                return """
The queries should use explicit OR logical combinations like:
- "parcels that are either [criterion A] OR [criterion B]"
- "properties meeting ANY of the following criteria: [list criteria]"

Each query should allow multiple alternative criteria for selection.
"""
            elif subcategory == "Nested Logical Structures":
                return """
The queries should use nested logical expressions like:
- "parcels that are [criterion A] AND (either [criterion B] OR [criterion C])"
- "properties that are either ([criterion A] AND [criterion B]) OR ([criterion C] AND [criterion D])"

Each query should use parentheses or clear language to create nested logical structures.
"""
            elif subcategory == "Conditional Structures":
                return """
The queries should use if-then conditions like:
- "parcels where, if [condition A], then [criterion B] applies, otherwise [criterion C] applies"
- "properties that must satisfy [criterion A] if they are [condition B], but must satisfy [criterion C] if they are [condition D]"

Each query should specify different criteria based on conditional properties.
"""

        # Default prompt section if not specified
        return f"""
Please create queries that fit the {category}/{subcategory} pattern. Make them realistic for commercial site selection in Cambridge, MA using ONLY the available data fields.
"""

    def _similarity(self, query1: str, query2: str) -> float:
        """
        Calculate simple similarity between two queries.

        Args:
            query1: First query
            query2: Second query

        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity for quick comparison
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0

    def _fix_column_references(self, code: str) -> str:
        """
        Fix incorrect column references in the code.
        This is critical to prevent KeyError exceptions.

        Args:
            code: Original code

        Returns:
            Fixed code with corrected column references
        """
        fixed_code = code

        # Apply all column corrections
        for wrong, correct in self.column_corrections.items():
            fixed_code = fixed_code.replace(wrong, correct)

        # Fix a specific error pattern: poi[poi['ml'] references
        fixed_code = fixed_code.replace(
            "poi[poi['poi_type']", "poi[poi['business_type']")
        fixed_code = fixed_code.replace(
            "poi_proj[poi_proj['poi_type']", "poi_proj[poi_proj['business_type']")

        fixed_code = fixed_code.replace(
            "poi[poi['ml']", "poi[poi['business_type']")
        fixed_code = fixed_code.replace(
            "poi_proj[poi_proj['ml']", "poi_proj[poi_proj['business_type']")

        return fixed_code

    def _fix_common_imports(self, code: str) -> str:
        """
        Fix common import errors without changing the visible code.

        Args:
            code: Original code

        Returns:
            Code with fixed imports for execution only
        """
        # Common imports to check for
        needed_imports = {
            "pd": "import pandas as pd",
            "np": "import numpy as np",
            "gpd": "import geopandas as gpd",
            "Point": "from shapely.geometry import Point",
            "LineString": "from shapely.geometry import LineString",
            "unary_union": "from shapely.ops import unary_union"
        }

        # Add any missing imports at the beginning
        imports_to_add = []

        for module_name, import_statement in needed_imports.items():
            # Check if module is used but not imported
            if module_name in code and import_statement not in code:
                imports_to_add.append(import_statement)

        # Add all missing imports at the beginning
        if imports_to_add:
            import_block = "\n".join(imports_to_add) + "\n\n"
            fixed_code = import_block + code
            return fixed_code

        return code

    def _fix_code_for_execution(self, code: str) -> str:
        """
        Apply multiple fixes to make code more likely to execute successfully.

        Args:
            code: Original code

        Returns:
            Fixed code for execution
        """
        # Fix imports
        fixed_code = self._fix_common_imports(code)

        # Fix column references
        fixed_code = self._fix_column_references(fixed_code)

        # Add try-except block for better error handling
        if "try:" not in fixed_code:
            indented_code = "\n    ".join(fixed_code.split("\n"))
            fixed_code = f"""try:
    {indented_code}
except Exception as e:
    print(f"Error: {{e}}")
    # Try to print partial results if available
    try:
        if 'result_parcels' in locals():
            print(result_parcels['ml'].tolist())
        elif 'final_parcels' in locals():
            print(final_parcels['ml'].tolist())
    except:
        print("No partial results available")
"""

        return fixed_code

    def generate_sample(self, query: str, category: str, subcategory: str) -> Dict[str, Any]:
        """
        Generate a sample with Query, Code, Answer (null if code has bugs).
        Automatically fixes common import errors.

        Args:
            query: The query string
            category: Constraint category
            subcategory: Constraint subcategory

        Returns:
            Dictionary with Query, Code, Answer fields
        """
        # Add dataset schema information to the prompt
        schema_prompt = """You are a commercial site selection assistant that creates Python code to analyze geospatial data in Cambridge, MA.
Generate executable GeoPandas code to find parcels matching the criteria.

AVAILABLE DATASETS - IMPORTANT COLUMN INFORMATION:
1. Parcels ('cambridge_parcels.geojson'):
   - 'ml': Parcel ID (string) - ALWAYS use this for the final result
   - 'use_code': Land use code (string) - NOT 'zoning' or 'land_use'
   - 'land_area': Size in square feet (numeric) - NOT 'area' or 'size'
   
2. POI ('cambridge_poi_processed.geojson'):
   - 'business_type': Type of business/POI - NOT 'category', 'type', or 'name'
   - 'geometry': Spatial location
   - 'PLACEKEY': Identifier for joining with spending data
   - NOTE: POI dataset does NOT have 'ml' column like parcels does
   
3. Census ('cambridge_census_cambridge_pct.geojson'):
   - 'pct_adv_deg': % with advanced degrees
   - 'pct_18_64': % aged 18-64
   (Note: No median_income field)

4. Spending ('cambridge_spend_processed.csv'):
   - 'PLACEKEY': Join key with POI data
   - 'RAW_TOTAL_SPEND': Consumer spending amount - NOT 'estimated_spend'

IMPORTANT: Always include all necessary imports at the beginning of your code."""

        # Create agent for code generation
        agent = SpatialAnalysisAgent(
            data_dir=self.data_dir,
            model_name=self.fine_tuned_model,
            openai_api_key=self.openai_api_key,
            system_prompt=schema_prompt
        )

        # Generate code for the query
        result = agent.run_conversation(query)

        # Always include the code, even if it has bugs
        if result.get("data") and result["data"].get("code"):
            original_code = result["data"]["code"]

            # Add fixes for common errors
            fixed_code = self._fix_code_for_execution(original_code)

            # Try to execute the code, but don't worry if it fails
            try:
                success, output, parcel_ids = agent.execute_code(fixed_code)

                # Create the sample dictionary
                sample = {
                    "Query": query,
                    "Code": original_code,  # Always keep the original code
                    "Answer": json.dumps(parcel_ids) if success else None,
                    "Category": category,
                    "Subcategory": subcategory
                }
            except Exception as e:
                logger.warning(f"Code execution error: {str(e)}")
                # Even with execution failure, still include the code with null answer
                sample = {
                    "Query": query,
                    "Code": original_code,  # Always keep the original code
                    "Answer": None,
                    "Category": category,
                    "Subcategory": subcategory
                }

            logger.info(
                f"Generated sample for {category}/{subcategory}: {query[:50]}...")
            return sample
        else:
            logger.warning(
                f"Failed to generate code for {category}/{subcategory}: {query}")
            return None

    def generate_samples_by_category(self, output_dir: str, counts: Dict[str, Dict[str, int]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate samples according to specified counts by category and subcategory.
        Saves category files first, then combines them at the end.

        Args:
            output_dir: Directory to save generated samples
            counts: Dictionary mapping categories to subcategories and counts

        Returns:
            Dictionary of generated samples by category
        """
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results container for each category
        category_samples = {}
        for category in counts.keys():
            category_samples[category] = []

        total_generated = 0

        # Generate samples for each category and subcategory
        for category, subcategories in counts.items():
            logger.info(f"Generating samples for category: {category}")

            for subcategory, count in subcategories.items():
                logger.info(f"  Subcategory: {subcategory}, Count: {count}")

                # Skip if count is 0
                if count <= 0:
                    continue

                # Generate queries for this subcategory
                queries = self.generate_structured_queries(
                    category, subcategory, count * 2)  # Generate extra for potential failures

                if not queries:
                    logger.warning(
                        f"No valid queries generated for {category}/{subcategory}")
                    continue

                # Generate samples for queries
                subcategory_samples = []
                successes = 0

                for query in tqdm(queries, desc=f"{category}/{subcategory}"):
                    # Stop if we've reached the target count
                    if successes >= count:
                        break

                    # Try to generate a sample
                    sample = self.generate_sample(query, category, subcategory)

                    # Add successful samples to our list
                    if sample:
                        # Store this sample for this category
                        category_samples[category].append(sample)
                        subcategory_samples.append(sample)
                        successes += 1
                        total_generated += 1

                    # Add a delay to avoid rate limits
                    time.sleep(2)

                logger.info(
                    f"Generated {len(subcategory_samples)}/{count} samples for {category}/{subcategory}")

            # Save category samples immediately after completion - THIS IS KEY
            if category_samples[category]:
                category_file = os.path.join(output_dir, f"{category}.json")
                with open(category_file, 'w') as f:
                    json.dump(category_samples[category], f, indent=2)
                logger.info(
                    f"Saved {len(category_samples[category])} samples to {category_file}")
            else:
                logger.warning(f"No samples generated for category {category}")

        # Now save all samples combined AFTER all categories are done
        all_samples = []
        for category_list in category_samples.values():
            all_samples.extend(category_list)

        if all_samples:  # Only save if we have samples
            all_samples_file = os.path.join(output_dir, "all_samples.json")
            with open(all_samples_file, 'w') as f:
                json.dump(all_samples, f, indent=2)
            logger.info(
                f"Saved {len(all_samples)} total samples to {all_samples_file}")
        else:
            logger.error("No samples were generated!")

        logger.info(f"Generated {total_generated} total samples")
        return category_samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured spatial analysis samples constrained to available datasets")

    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Directory containing the data files")
    parser.add_argument("--model", type=str, required=True,
                        help="Fine-tuned model ID to use for generation")
    parser.add_argument("--output_dir", type=str, default="../generated_samples",
                        help="Directory to save generated samples")
    parser.add_argument("--simple_count", type=int, default=100,
                        help="Number of simple constraint samples to generate")
    parser.add_argument("--complex_count", type=int, default=100,
                        help="Number of complex constraint samples to generate")
    parser.add_argument("--spatial_count", type=int, default=50,
                        help="Number of spatial constraint samples to generate")
    parser.add_argument("--business_count", type=int, default=50,
                        help="Number of business environment constraint samples to generate")
    parser.add_argument("--demographic_count", type=int, default=50,
                        help="Number of demographic constraint samples to generate")
    parser.add_argument("--logical_count", type=int, default=50,
                        help="Number of logical structure constraint samples to generate")

    args = parser.parse_args()

    # Initialize dataset-constrained sample generator
    generator = DataConstrainedSampleGenerator(
        data_dir=args.data_dir,
        fine_tuned_model=args.model
    )

    # Set up sample counts by category and subcategory
    sample_counts = {
        "Simple_Constraints": {
            "Single Hard Constraint": args.simple_count // 4 + (args.simple_count % 4 > 0),
            "Single Soft Constraint": args.simple_count // 4 + (args.simple_count % 4 > 1),
            "Two Hard Constraints": args.simple_count // 4 + (args.simple_count % 4 > 2),
            "One Hard + One Soft Constraint": args.simple_count // 4
        },
        "Complex_Constraints": {
            "Multiple Hard Constraints": args.complex_count // 4 + (args.complex_count % 4 > 0),
            "Multiple Mixed Constraints": args.complex_count // 4 + (args.complex_count % 4 > 1),
            "Logical Combinations": args.complex_count // 4 + (args.complex_count % 4 > 2),
            "Conditional Constraints": args.complex_count // 4
        },
        "Spatial_Constraints": {
            "Simple Buffer Queries": args.spatial_count // 2 + (args.spatial_count % 2),
            "Nested Spatial Relationships": args.spatial_count // 2
        },
        "Business_Environment_Constraints": {
            "Competitor Density": args.business_count // 3 + (args.business_count % 3 > 0),
            "Land Use Mix": args.business_count // 3 + (args.business_count % 3 > 1),
            "Consumer Spending Patterns": args.business_count // 3
        },
        "Demographic_Constraints": {
            "Income Levels": args.demographic_count // 2 + (args.demographic_count % 2),
            "Target Demographic Match": args.demographic_count // 2
        },
        "Logical_Structure_Constraints": {
            "AND Combinations": args.logical_count // 4 + (args.logical_count % 4 > 0),
            "OR Combinations": args.logical_count // 4 + (args.logical_count % 4 > 1),
            "Nested Logical Structures": args.logical_count // 4 + (args.logical_count % 4 > 2),
            "Conditional Structures": args.logical_count // 4
        }
    }

    # Generate samples by category
    generator.generate_samples_by_category(args.output_dir, sample_counts)

    logger.info("Sample generation complete!")
    logger.info(f"Generated samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()

# # python structured_sample_generator.py \
#   --model ft:gpt-4o-mini-2024-07-18:mit:spatial-agent-20250505-021001:BTjW7EkG \
#   --data_dir data \
#   --output_dir generated_samples \
#   --simple_count 55 \
#   --complex_count 55 \
#   --spatial_count 30 \
#   --business_count 30 \
#   --demographic_count 20 \
#   --logical_count 20

# # python structured_sample_generator.py \
#   --model ft:gpt-4o-mini-2024-07-18:mit:spatial-agent-20250505-021001:BTjW7EkG \
#   --data_dir data \
#   --output_dir generated_samples \
#   --simple_count 2 \
#   --complex_count 2 \
#   --spatial_count 1 \
#   --business_count 1 \
#   --demographic_count 1 \
#   --logical_count 1
