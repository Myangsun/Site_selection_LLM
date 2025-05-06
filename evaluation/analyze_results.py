import os
import json
import argparse
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_results_only(args):
    """Run only the result analysis part."""

    # Create required directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Step: Analyze results
    logger.info("Analyzing results...")
    from result_analyzer import ResultAnalyzer

    # Initialize analyzer
    analyzer = ResultAnalyzer(args.results_dir)

    # Generate comprehensive report
    summary_df = analyzer.generate_report(
        os.path.join(args.output_dir, "analysis"))

    logger.info(
        f"Analysis complete. Report generated in {args.output_dir}/analysis")

    # Print summary
    print("\nEvaluation Summary:")
    print(summary_df)

    print("\nAnalysis complete!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Analyze results from spatial analysis method evaluations")

    # General arguments
    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory containing the evaluation results")
    parser.add_argument("--output_dir", type=str, default="../results/analysis",
                        help="Directory to store analysis results")

    args = parser.parse_args()

    # Run analysis only
    analyze_results_only(args)


if __name__ == "__main__":
    main()
