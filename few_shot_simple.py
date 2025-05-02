#!/usr/bin/env python
"""
Few-Shot Learning Method for Spatial Analysis (Simplified)

This script evaluates spatial analysis code generation using few-shot learning.
It treats all samples uniformly without categorization and uses direct comparison.

Usage:
    python few_shot_simple.py --samples-file spatial_samples.json --api-key YOUR_API_KEY
"""

import os
import json
import time
import argparse
import random
from tqdm import tqdm

try:
    import openai
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai

# Import the simple evaluator
from evaluator import Evaluator


def run_few_shot_evaluation(samples_file, api_key, output_dir="results", model="gpt-4o",
                            num_examples=3, verbose=True):
    """
    Run few-shot evaluation on all samples.

    Args:
        samples_file: Path to JSON file with samples
        api_key: OpenAI API key
        output_dir: Directory for saving results
        model: Model to use for evaluation
        num_examples: Number of examples to include in prompts
        verbose: Whether to print detailed progress

    Returns:
        Tuple of (results, summary)
    """
    # Initialize the evaluator
    evaluator = Evaluator(output_dir=output_dir, verbose=verbose)

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Load samples
    with open(samples_file, 'r', encoding='utf-8-sig') as f:
        all_samples = json.load(f)

    if verbose:
        print(f"Loaded {len(all_samples)} samples")
        print(f"Running {num_examples}-Shot Evaluation with {model}")

    results = []
    total_tokens = 0
    total_time = 0

    # Process each sample
    for i, sample in enumerate(tqdm(all_samples, desc="Evaluating samples", disable=not verbose)):
        # Select examples (avoiding the current sample)
        examples = select_examples(all_samples, sample, num_examples)

        # Create prompt with examples
        prompt = create_few_shot_prompt(sample, examples)

        # Track time and token usage
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2  # Lower temperature for more deterministic code generation
            )
            end_time = time.time()

            # Extract generated code
            generated_code = response.choices[0].message.content

            # Get reference answer
            reference_answer = sample.get("Answer", "")

            # Evaluate response using the simple evaluator
            evaluation = evaluator.evaluate_response(
                generated_code=generated_code,
                reference_answer=reference_answer
            )

            # Add performance metrics
            evaluation["response_time"] = end_time - start_time
            evaluation["tokens_used"] = response.usage.total_tokens
            evaluation["prompt_tokens"] = response.usage.prompt_tokens
            evaluation["completion_tokens"] = response.usage.completion_tokens

            # Update totals
            total_tokens += response.usage.total_tokens
            total_time += (end_time - start_time)

        except Exception as e:
            if verbose:
                print(f"Error evaluating sample {i}: {e}")
            # Create an empty evaluation
            evaluation = {
                "result_match_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "code_correctness": 0.0,
                "response_time": 0.0,
                "tokens_used": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
            generated_code = "Error: Failed to generate code"

        # Store result
        result = {
            "sample": sample,
            "examples": examples,
            "generated_code": generated_code,
            "evaluation": evaluation
        }
        results.append(result)

    # Compute average metrics
    summary = evaluator.compute_average_metrics(results)

    # Add overall performance metrics
    summary["total_tokens"] = total_tokens
    summary["total_time"] = total_time
    summary["avg_tokens_per_sample"] = total_tokens / \
        len(all_samples) if all_samples else 0
    summary["avg_time_per_sample"] = total_time / \
        len(all_samples) if all_samples else 0

    # Save results
    evaluator.save_results("few_shot", results, summary)

    if verbose:
        print(f"Few-shot evaluation complete")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {total_time:.2f}s")
        print(
            f"Average tokens per sample: {summary['avg_tokens_per_sample']:.1f}")
        print(
            f"Average time per sample: {summary['avg_time_per_sample']:.2f}s")
        print(f"Average F1 score: {summary['f1_score']:.4f}")

    return results, summary


def select_examples(all_samples, current_sample, num_examples):
    """
    Select random examples for few-shot learning, avoiding the current sample.

    Args:
        all_samples: List of all available samples
        current_sample: The sample being evaluated
        num_examples: Number of examples to select

    Returns:
        List of selected example samples
    """
    # Filter out the current sample
    available_samples = [s for s in all_samples if s != current_sample]

    # If we have fewer samples than requested examples, use all available
    if len(available_samples) <= num_examples:
        return available_samples

    # Otherwise, randomly select the requested number of examples
    return random.sample(available_samples, num_examples)


def create_few_shot_prompt(sample, examples):
    """
    Create a few-shot prompt with examples.

    Args:
        sample: The sample to evaluate
        examples: List of example samples to include

    Returns:
        Formatted prompt string
    """
    prompt = "Here are some examples of spatial analysis queries and their corresponding code:\n\n"

    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Query: {example['Query']}\n\n"
        prompt += f"Code:\n{example['Code']}\n\n"
        prompt += f"Answer: {example['Answer']}\n\n"
        prompt += f"{'='*50}\n\n"

    prompt += f"Now, generate Python code for this new spatial analysis query:\n\n{sample['Query']}\n\n"
    prompt += "Provide just the Python code without explanations."

    return prompt


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Few-Shot Learning Evaluation for Spatial Analysis")
    parser.add_argument("--samples-file", required=True,
                        help="Path to samples JSON file")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for results")
    parser.add_argument("--model", default="gpt-4o",
                        help="Model to use for evaluation")
    parser.add_argument("--num-examples", type=int, default=3,
                        help="Number of examples to include")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()

    # Run evaluation
    run_few_shot_evaluation(
        samples_file=args.samples_file,
        api_key=args.api_key,
        output_dir=args.output_dir,
        model=args.model,
        num_examples=args.num_examples,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
