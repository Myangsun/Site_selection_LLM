import os
import json
import logging
import random
import time
from typing import List, Dict, Any, Optional, Tuple
import argparse
from tqdm import tqdm
from fine_tuned_rag_evaluator import SpatialRAGComponent
from implementation.agent_framework import SpatialAnalysisAgent

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StructuredSampleGenerator:
    """Generate structured samples for spatial analysis queries."""

    def __init__(self, data_dir: str, fine_tuned_model: str, openai_api_key: Optional[str] = None):
        """
        Initialize the structured sample generator.

        Args:
            data_dir: Directory containing the geospatial datasets
            fine_tuned_model: Fine-tuned model to use for generation
            openai_api_key: OpenAI API key for API access
        """
        self.data_dir = data_dir
        self.fine_tuned_model = fine_tuned_model
        self.openai_api_key = openai_api_key or os.environ.get(
            "OPENAI_API_KEY")

        # Initialize RAG component
        self.rag_component = SpatialRAGComponent(data_dir)

        # Load existing samples
        self.existing_samples = self._load_existing_samples()

        # Set up data file paths
        self.data_files = {
            'parcels': os.path.join(data_dir, 'cambridge_parcels.geojson'),
            'poi': os.path.join(data_dir, 'cambridge_poi_processed.geojson'),
            'census': os.path.join(data_dir, 'cambridge_census_cambridge_pct.geojson'),
            'spend': os.path.join(data_dir, 'cambridge_spend_processed.csv')
        }

        logger.info(
            f"Structured sample generator initialized with {len(self.existing_samples)} existing samples")

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
        Generate structured queries for a specific category and subcategory.

        Args:
            category: Main constraint category
            subcategory: Subcategory within the main category
            count: Number of queries to generate

        Returns:
            List of generated query strings
        """
        # Create detailed prompt for the specified category and subcategory
        prompt = self._create_category_prompt(category, subcategory, count)

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

                # Filter out queries that are too similar to existing queries
                existing_queries = [sample["Query"]
                                    for sample in self.existing_samples]
                filtered_queries = [
                    q for q in queries
                    if not any(self._similarity(q, eq) > 0.8 for eq in existing_queries)
                ]

                logger.info(
                    f"Generated {len(filtered_queries)} unique {category}/{subcategory} queries")
                return filtered_queries[:count]  # Limit to requested count
            else:
                logger.error("Could not find JSON array in response")
                return []
        except Exception as e:
            logger.error(f"Error extracting queries: {e}")
            return []

    def _create_category_prompt(self, category: str, subcategory: str, count: int) -> str:
        """
        Create a detailed prompt for generating queries in a specific category.

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
            existing_queries, min(5, len(existing_queries)))

        # Basic prompt structure
        base_prompt = f"""You are an expert in geospatial analysis and site selection for commercial real estate.

Here are some examples of site selection queries:
{json.dumps(example_queries, indent=2)}

I need you to create {count} new queries specifically for the category: "{category}" and subcategory: "{subcategory}"."""

        # Add category-specific details
        if category == "Simple Constraints":
            if subcategory == "Single Hard Constraint":
                return base_prompt + """
The queries should have exactly ONE hard constraint like:
- Size requirements (e.g., "larger than X square feet")
- Specific zoning/use code (e.g., "zoned for retail")
- Exact location criteria (e.g., "within X meters of [location]")

Each query should focus on finding parcels with ONE clear, non-negotiable requirement.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Single Soft Constraint":
                return base_prompt + """
The queries should have exactly ONE soft constraint like:
- Prioritizing certain areas (e.g., "prioritizing areas with higher foot traffic")
- Preference criteria (e.g., "preferably near residential neighborhoods")
- Optimization goals (e.g., "with the highest consumer spending")

Each query should use language indicating preference rather than requirement (e.g., "prioritizing," "preferably," "ideally").

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Two Hard Constraints":
                return base_prompt + """
The queries should have exactly TWO hard constraints combined with "and" like:
- "larger than X square feet AND within Y meters of [location]"
- "zoned for retail AND with at least Z parking spaces"

Each query should have two clear, non-negotiable requirements that both must be satisfied.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "One Hard + One Soft Constraint":
                return base_prompt + """
The queries should combine ONE hard constraint and ONE soft constraint like:
- "larger than X square feet, preferably in areas with high foot traffic"
- "within Y meters of subway stations, prioritizing areas with higher income levels"

Each query should have one clear requirement and one preference criterion.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

        elif category == "Complex Constraints":
            if subcategory == "Multiple Hard Constraints":
                return base_prompt + """
The queries should have THREE OR MORE hard constraints like:
- "larger than X square feet, within Y meters of subway stations, and with Z parking spaces"
- "zoned for retail, not within 500m of competitors, and in census tracts with median income above $40,000"

Each query should have at least three non-negotiable requirements that all must be satisfied.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Multiple Mixed Constraints":
                return base_prompt + """
The queries should have at least TWO hard constraints AND TWO soft constraints like:
- "larger than X square feet and within Y meters of subway stations, preferably in areas with high foot traffic and low competition"
- "zoned for retail and with Z parking spaces, ideally in areas with high educational attainment and near residential neighborhoods"

Each query should clearly distinguish between requirements and preferences.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Logical Combinations":
                return base_prompt + """
The queries should use complex logical combinations (AND, OR, NOT) like:
- "either office space larger than X sq ft OR retail space within Y meters of residential areas"
- "commercial parcels NOT within Z meters of industrial zones AND either near transit OR with high foot traffic"

Each query should use logical operators to create complex selection criteria.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Conditional Constraints":
                return base_prompt + """
The queries should use conditional (if-then) logic like:
- "parcels that, if larger than X sq ft, must be within Y meters of transit, or if smaller, must be within Z meters of residential areas"
- "mixed-use parcels that, if north of [location], must have retail on ground floor, otherwise must have at least X sq ft of commercial space"

Each query should specify different criteria depending on some property of the parcel.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

        elif category == "Spatial Constraints":
            if subcategory == "Simple Buffer Queries":
                return base_prompt + """
The queries should use simple buffer/distance criteria like:
- "within X meters of [location/feature]"
- "at least Y meters away from [location/feature]"
- "between X and Y meters from [location/feature]"

Each query should specify a clear spatial relationship based on distance.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Nested Spatial Relationships":
                return base_prompt + """
The queries should use nested or complex spatial relationships like:
- "within X meters of [feature A] AND outside Y meters of [feature B]"
- "parcels that intersect with buffer zones of [feature A] but are at least Z meters from [feature B]"
- "within X meters of [feature A] OR within Y meters of [feature B], but not within Z meters of [feature C]"

Each query should combine multiple spatial relationships in a nested or complex manner.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

        elif category == "Business Environment Constraints":
            if subcategory == "Competitor Density":
                return base_prompt + """
The queries should focus on competitor density/proximity like:
- "with no more than X competing businesses within Y meters"
- "at least Z meters away from nearest competitor of same type"
- "in areas with the lowest density of competing [business type]"

Each query should specify criteria related to the presence or absence of competing businesses.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Land Use Mix":
                return base_prompt + """
The queries should focus on the mix of surrounding land uses like:
- "in areas with diverse mix of residential and commercial uses within X meters"
- "surrounded by at least Y different land use types within Z meters"
- "in predominantly [land use type] areas but with some [other land use type] nearby"

Each query should specify criteria related to the diversity or composition of surrounding land uses.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Consumer Spending Patterns":
                return base_prompt + """
The queries should focus on consumer spending patterns like:
- "in areas with highest average consumer spending"
- "where spending on [category] is at least X% above city average"
- "top Y parcels ranked by total consumer spending within Z meters"

Each query should specify criteria related to consumer spending levels or patterns.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

        elif category == "Demographic Constraints":
            if subcategory == "Income Levels":
                return base_prompt + """
The queries should focus on income-related demographics like:
- "in census tracts with median income above $X"
- "in areas where at least Y% of households earn more than $Z"
- "ranked by neighborhood affluence level"

Each query should specify criteria related to income levels of the surrounding population.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Target Demographic Match":
                return base_prompt + """
The queries should focus on specific demographic targets like:
- "in areas with at least X% of residents aged 25-34"
- "where percentage of residents with advanced degrees is above Y%"
- "in neighborhoods with highest concentration of [demographic group]"

Each query should specify criteria related to the demographic composition of the surrounding population.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

        elif category == "Logical Structure Constraints":
            if subcategory == "AND Combinations":
                return base_prompt + """
The queries should use explicit AND logical combinations like:
- "parcels that are [criterion A] AND [criterion B] AND [criterion C]"
- "properties meeting ALL of the following criteria: [list criteria]"

Each query should require that multiple criteria be satisfied simultaneously.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "OR Combinations":
                return base_prompt + """
The queries should use explicit OR logical combinations like:
- "parcels that are either [criterion A] OR [criterion B]"
- "properties meeting ANY of the following criteria: [list criteria]"

Each query should allow multiple alternative criteria for selection.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Nested Logical Structures":
                return base_prompt + """
The queries should use nested logical expressions like:
- "parcels that are [criterion A] AND (either [criterion B] OR [criterion C])"
- "properties that are either ([criterion A] AND [criterion B]) OR ([criterion C] AND [criterion D])"

Each query should use parentheses or clear language to create nested logical structures.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

            elif subcategory == "Conditional Structures":
                return base_prompt + """
The queries should use if-then conditions like:
- "parcels where, if [condition A], then [criterion B] applies, otherwise [criterion C] applies"
- "properties that must satisfy [criterion A] if they are [condition B], but must satisfy [criterion C] if they are [condition D]"

Each query should specify different criteria based on conditional properties.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

        # Default case
        return base_prompt + f"""
Please create {count} diverse queries that fit the {category}/{subcategory} pattern. Make them realistic for commercial site selection in Cambridge, MA.

Return ONLY a JSON array of query strings, with no explanations or additional text."""

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

    def generate_sample(self, query: str, category: str, subcategory: str) -> Dict[str, Any]:
        """
        Generate a complete sample (Query, Code, Answer) for a given query.

        Args:
            query: The query string
            category: Constraint category
            subcategory: Constraint subcategory

        Returns:
            Dictionary with Query, Code, Answer fields
        """
        # Enhance query with RAG
        enhanced_query = self.rag_component.enhance_prompt(query)

        # System prompt for code generation
        system_prompt = """You are a commercial site selection assistant that creates Python code to analyze geospatial data in Cambridge, MA.
Generate executable GeoPandas code to find parcels matching the criteria."""

        # Create agent for code generation
        agent = SpatialAnalysisAgent(
            data_dir=self.data_dir,
            model_name=self.fine_tuned_model,
            openai_api_key=self.openai_api_key,
            system_prompt=system_prompt
        )

        # Generate code for the query
        result = agent.run_conversation(enhanced_query)

        # Check if code was generated and executed successfully
        if (result.get("data") and
            result["data"].get("code") and
                result["data"].get("parcel_ids")):

            code = result["data"]["code"]
            parcel_ids = result["data"]["parcel_ids"]

            # Create the sample dictionary
            sample = {
                "Query": query,
                "Code": code,
                "Answer": json.dumps(parcel_ids),
                "Category": category,
                "Subcategory": subcategory
            }

            logger.info(
                f"Successfully generated sample for {category}/{subcategory}: {query[:50]}...")
            return sample
        else:
            logger.warning(
                f"Failed to generate complete sample for {category}/{subcategory}: {query}")
            return None

    def generate_samples_by_category(self, output_dir: str, counts: Dict[str, Dict[str, int]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate samples according to specified counts by category and subcategory.

        Args:
            output_dir: Directory to save generated samples
            counts: Dictionary mapping categories to subcategories and counts

        Returns:
            Dictionary of generated samples by category
        """
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results container
        all_samples = {}
        total_generated = 0

        # Generate samples for each category and subcategory
        for category, subcategories in counts.items():
            category_samples = []

            # Create directory for category
            category_dir = os.path.join(output_dir, category.replace(" ", "_"))
            os.makedirs(category_dir, exist_ok=True)

            logger.info(f"Generating samples for category: {category}")

            for subcategory, count in subcategories.items():
                logger.info(f"  Subcategory: {subcategory}, Count: {count}")

                # Skip if count is 0
                if count <= 0:
                    continue

                # Generate queries for this subcategory
                queries = self.generate_structured_queries(
                    category, subcategory, count * 2)  # Generate extra for potential failures

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
                        subcategory_samples.append(sample)
                        category_samples.append(sample)
                        successes += 1
                        total_generated += 1

                    # Add a delay to avoid rate limits
                    time.sleep(2)

                # Save subcategory samples
                subcategory_file = os.path.join(
                    category_dir, f"{subcategory.replace(' ', '_')}.json")
                with open(subcategory_file, 'w') as f:
                    json.dump(subcategory_samples, f, indent=2)

                logger.info(
                    f"Generated {len(subcategory_samples)}/{count} samples for {category}/{subcategory}")

            # Save category samples
            category_file = os.path.join(
                output_dir, f"{category.replace(' ', '_')}.json")
            with open(category_file, 'w') as f:
                json.dump(category_samples, f, indent=2)

            all_samples[category] = category_samples

        # Save all samples
        all_samples_file = os.path.join(output_dir, "all_samples.json")
        with open(all_samples_file, 'w') as f:
            json.dump([sample for category_samples in all_samples.values()
                      for sample in category_samples], f, indent=2)

        logger.info(f"Generated {total_generated} total samples")
        return all_samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured spatial analysis samples")

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

    # Initialize sample generator
    generator = StructuredSampleGenerator(
        data_dir=args.data_dir,
        fine_tuned_model=args.model
    )

    # Set up sample counts by category and subcategory
    sample_counts = {
        "Simple Constraints": {
            "Single Hard Constraint": args.simple_count // 4 + (args.simple_count % 4 > 0),
            "Single Soft Constraint": args.simple_count // 4 + (args.simple_count % 4 > 1),
            "Two Hard Constraints": args.simple_count // 4 + (args.simple_count % 4 > 2),
            "One Hard + One Soft Constraint": args.simple_count // 4
        },
        "Complex Constraints": {
            "Multiple Hard Constraints": args.complex_count // 4 + (args.complex_count % 4 > 0),
            "Multiple Mixed Constraints": args.complex_count // 4 + (args.complex_count % 4 > 1),
            "Logical Combinations": args.complex_count // 4 + (args.complex_count % 4 > 2),
            "Conditional Constraints": args.complex_count // 4
        },
        "Spatial Constraints": {
            "Simple Buffer Queries": args.spatial_count // 2 + (args.spatial_count % 2),
            "Nested Spatial Relationships": args.spatial_count // 2
        },
        "Business Environment Constraints": {
            "Competitor Density": args.business_count // 3 + (args.business_count % 3 > 0),
            "Land Use Mix": args.business_count // 3 + (args.business_count % 3 > 1),
            "Consumer Spending Patterns": args.business_count // 3
        },
        "Demographic Constraints": {
            "Income Levels": args.demographic_count // 2 + (args.demographic_count % 2),
            "Target Demographic Match": args.demographic_count // 2
        },
        "Logical Structure Constraints": {
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
#   --data_dir ../data \
#   --output_dir ../generated_samples \
#   --simple_count 100 \
#   --complex_count 100 \
#   --spatial_count 50 \
#   --business_count 50 \
#   --demographic_count 50 \
#   --logical_count 50
