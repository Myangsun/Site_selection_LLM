#!/usr/bin/env python3

import os
import json
import argparse
import logging
import random
from typing import Dict, Any, List
import subprocess
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import openai
        import pandas
        import geopandas
        import matplotlib
        import seaborn
        import tqdm
        import shapely
        logger.info("All required dependencies are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Installing required dependencies...")
        subprocess.run([
            "pip", "install",
            "openai", "pandas", "geopandas", "matplotlib", "seaborn",
            "tqdm", "shapely"
        ])
        return False


def run_method_evaluation(args):
    """Run the method evaluation."""
    # Create required directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test samples
    sample_path = os.path.join(args.data_dir, 'spatial_samples.json')
    if not os.path.exists(sample_path):
        logger.error(f"Test samples file not found at {sample_path}")
        return False

    with open(sample_path, 'r', encoding='utf-8-sig') as f:
        test_samples = json.load(f)

    logger.info(f"Loaded {len(test_samples)} test samples")

    # Set up data files
    data_files = {
        'parcels': os.path.join(args.data_dir, 'cambridge_parcels.geojson'),
        'poi': os.path.join(args.data_dir, 'cambridge_poi_processed.geojson'),
        'census': os.path.join(args.data_dir, 'cambridge_census_cambridge_pct.geojson'),
        'spend': os.path.join(args.data_dir, 'cambridge_spend_processed.csv')
    }

    # Prepare sample subset if requested
    if args.sample_size and args.sample_size < len(test_samples):
        samples = random.sample(test_samples, args.sample_size)
        logger.info(f"Using {args.sample_size} samples for evaluation")
    else:
        samples = test_samples
        logger.info(f"Using all {len(samples)} samples for evaluation")

    # Step 1: Run zero-shot evaluation
    if "zero-shot" in args.methods:
        logger.info("Running zero-shot evaluation...")
        from zero_shot_evaluator import ZeroShotEvaluator

        # Initialize evaluator
        zero_shot = ZeroShotEvaluator(args.data_dir, test_samples, data_files)

        # Run evaluation
        results = zero_shot.evaluate(samples)

        # Save results
        with open(os.path.join(args.output_dir, "zero_shot_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Zero-shot evaluation complete. Results saved to {args.output_dir}/zero_shot_results.json")

    # Step 2: Run few-shot evaluation
    if "few-shot" in args.methods:
        logger.info(f"Running few-shot evaluation with 5 examples...")
        from few_shot_evaluator import FewShotEvaluator

        # Initialize evaluator
        few_shot = FewShotEvaluator(args.data_dir, test_samples, data_files)

        # Run evaluation
        results = few_shot.evaluate(samples, num_examples=5)

        # Save results
        with open(os.path.join(args.output_dir, "few_shot_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Few-shot evaluation complete. Results saved to {args.output_dir}/few_shot_results.json")

    # Step 3: Run fine-tuning if requested
    if "fine-tuned" in args.methods:
        # Check if we should skip fine-tuning
        if args.skip_finetune:
            # Try to get fine-tuned model from environment variable
            fine_tuned_model = os.environ.get("FINE_TUNED_MODEL_NAME")

            if fine_tuned_model:
                logger.info(
                    f"Skipping fine-tuning, using existing model: {fine_tuned_model}")

                # Skip to evaluation with the existing model
                from fine_tuned_evaluator import FineTunedEvaluator

                # Initialize evaluator
                fine_tuned_eval = FineTunedEvaluator(
                    args.data_dir, test_samples, data_files)

                # Run evaluation
                results = fine_tuned_eval.evaluate(samples, fine_tuned_model)

                # Save results
                with open(os.path.join(args.output_dir, "fine_tuned_results.json"), 'w') as f:
                    json.dump(results, f, indent=2)

                logger.info(
                    f"Fine-tuned model evaluation complete. Results saved to {args.output_dir}/fine_tuned_results.json")
            else:
                logger.error(
                    "No fine-tuned model found. Set the FINE_TUNED_MODEL_NAME environment variable.")
        else:

            # ALWAYS use multi-turn format
            logger.info(
                "Running fine-tuning with multi-turn conversation format...")
            from finetune_multi_turn import SpatialFineTuningHandler

            # Initialize fine-tuner
            fine_tuner = SpatialFineTuningHandler(args.data_dir)

            # Run fine-tuning pipeline
            fine_tuning_result = fine_tuner.run_fine_tuning_pipeline(
                output_dir=os.path.join(args.output_dir, "finetune_data"),
                model="gpt-4o-2024-08-06",
                validation_split=0.2,
                wait_for_completion=True
            )

            # Save fine-tuning result
            with open(os.path.join(args.output_dir, "fine_tuning_result.json"), 'w') as f:
                # Convert non-serializable objects to strings
                serializable_result = {}
                for key, value in fine_tuning_result.items():
                    if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                        serializable_result[key] = value
                    else:
                        serializable_result[key] = str(value)

                json.dump(serializable_result, f, indent=2)

            logger.info(
                f"Fine-tuning information saved to {args.output_dir}/fine_tuning_result.json")

            # Get fine-tuned model if available
            fine_tuned_model = None
            if isinstance(fine_tuning_result, dict) and 'status' in fine_tuning_result:
                if isinstance(fine_tuning_result['status'], dict):
                    fine_tuned_model = fine_tuning_result['status'].get(
                        'fine_tuned_model')
                elif isinstance(fine_tuning_result['status'], str) and fine_tuning_result['status'] == "succeeded":
                    # If status is just a string success indicator, check for model directly
                    fine_tuned_model = fine_tuning_result.get(
                        'fine_tuned_model')

            if fine_tuned_model:
                # Store the model ID for later use
                os.environ["FINE_TUNED_MODEL_NAME"] = fine_tuned_model

                logger.info(
                    f"Running evaluation with fine-tuned model: {fine_tuned_model}")
                from fine_tuned_evaluator import FineTunedEvaluator

                # Initialize evaluator
                fine_tuned_eval = FineTunedEvaluator(
                    args.data_dir, test_samples, data_files)

                # Run evaluation
                results = fine_tuned_eval.evaluate(samples, fine_tuned_model)

                # Save results
                with open(os.path.join(args.output_dir, "fine_tuned_results.json"), 'w') as f:
                    json.dump(results, f, indent=2)

                logger.info(
                    f"Fine-tuned model evaluation complete. Results saved to {args.output_dir}/fine_tuned_results.json")

                # Print the model ID for convenience
                print(f"\nFine-tuned model ID: {fine_tuned_model}")
                print("Use this ID with the agent implementation.")

    # Step 4: Run fine-tuned RAG evaluation
    if "fine-tuned-rag" in args.methods:
        logger.info(f"Running fine-tuned RAG evaluation...")
        from fine_tuned_rag_evaluator import FineTunedRAGEvaluator

        # Get fine-tuned model if available
        fine_tuned_model = os.environ.get("FINE_TUNED_MODEL_NAME")

        if fine_tuned_model:
            # Initialize evaluator
            fine_tuned_rag_eval = FineTunedRAGEvaluator(
                args.data_dir, test_samples, data_files)

            # Run evaluation
            results = fine_tuned_rag_eval.evaluate(samples, fine_tuned_model)

            # Save results
            with open(os.path.join(args.output_dir, "fine_tuned_rag_results.json"), 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(
                f"Fine-tuned RAG evaluation complete. Results saved to {args.output_dir}/fine_tuned_rag_results.json")
        else:
            logger.warning("No fine-tuned model available for RAG evaluation")


# Step 5: Analyze results
    logger.info("Analyzing results...")
    from result_analyzer import ResultAnalyzer

    # Initialize analyzer
    analyzer = ResultAnalyzer(args.output_dir)

    # Generate comprehensive report
    summary_df = analyzer.generate_report(
        os.path.join(args.output_dir, "analysis"))

    logger.info(
        f"Analysis complete. Report generated in {args.output_dir}/analysis")

    # Print summary
    print("\nEvaluation Summary:")
    print(summary_df)

    print("\nMethod evaluation complete!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate different methods for spatial analysis")

    # General arguments
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Directory containing the data files")
    parser.add_argument("--output_dir", type=str, default="../results",
                        help="Directory to store evaluation results")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to use for evaluation")

    # Methods to evaluate (simplified)
    parser.add_argument("--methods", nargs="+", default=["zero-shot", "few-shot", "fine-tuned"],
                        choices=["zero-shot", "few-shot",
                                 "fine-tuned", "fine-tuned-rag"],
                        help="Methods to evaluate")

    # Add to the argument parser in evaluate_methods.py
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip the fine-tuning step and use existing model for evaluation")

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Run method evaluation
    run_method_evaluation(args)


if __name__ == "__main__":
    main()
