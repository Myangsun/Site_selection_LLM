#!/usr/bin/env python
"""
Fine-Tuned Model Method for Spatial Analysis (Simplified)

Usage:
    python fine_tuned_simple.py --samples-file spatial_samples.json 
                               --api-key YOUR_API_KEY 
                               --fine-tuned-model YOUR_FINE_TUNED_MODEL_ID
"""

import os
import json
import time
import argparse
from tqdm import tqdm

import openai
from evaluator import Evaluator


def run_fine_tuned_evaluation(samples_file, api_key, fine_tuned_model,
                              output_dir="results", verbose=True):
    """Run evaluation with the fine-tuned model on all samples."""
    evaluator = Evaluator(output_dir=output_dir, verbose=verbose)
    client = openai.OpenAI(api_key=api_key)

    # Load samples
    with open(samples_file, 'r') as f:
        samples = json.load(f)

    if verbose:
        print(f"Loaded {len(samples)} samples")
        print(f"Running evaluation with fine-tuned model: {fine_tuned_model}")

    results = []
    total_tokens = 0
    total_time = 0.0

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating samples", disable=not verbose)):
        prompt = f"Generate Python code for this spatial analysis query: {sample['Query']}"
        start = time.time()

        try:
            resp = client.chat.completions.create(
                model=fine_tuned_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            elapsed = time.time() - start

            code = resp.choices[0].message.content
            reference = sample.get("Answer", "")

            eval_metrics = evaluator.evaluate_response(
                generated_code=code,
                reference_answer=reference
            )
            eval_metrics.update({
                "response_time": elapsed,
                "tokens_used": resp.usage.total_tokens,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            })

            total_tokens += resp.usage.total_tokens
            total_time += elapsed

        except Exception as e:
            if verbose:
                print(f"[Error] Sample {idx}: {e}")
            eval_metrics = {
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
            code = f"# Error generating code: {e}"

        results.append({
            "sample": sample,
            "generated_code": code,
            "evaluation": eval_metrics
        })

    summary = evaluator.compute_average_metrics(results)
    summary.update({
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tokens_per_sample": total_tokens / len(samples) if samples else 0,
        "avg_time_per_sample": total_time / len(samples) if samples else 0
    })

    evaluator.save_results("fine_tuned", results, summary)

    if verbose:
        print("=== Evaluation Complete ===")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg tokens/sample: {summary['avg_tokens_per_sample']:.1f}")
        print(f"Avg time/sample: {summary['avg_time_per_sample']:.2f}s")
        print(f"Avg F1 score: {summary['f1_score']:.4f}")

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Fine-Tuned Model Evaluation for Spatial Analysis"
    )
    parser.add_argument("--samples-file", required=True,
                        help="Path to samples JSON file")
    parser.add_argument("--api-key", required=True,
                        help="OpenAI API key")
    parser.add_argument("--fine-tuned-model", required=True,
                        help="ID of the fine-tuned model")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for saving results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    run_fine_tuned_evaluation(
        samples_file=args.samples_file,
        api_key=args.api_key,
        fine_tuned_model=args.fine_tuned_model,
        output_dir=args.output_dir,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
