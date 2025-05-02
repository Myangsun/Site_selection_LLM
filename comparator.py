#!/usr/bin/env python
"""
Simple Results Comparator for Spatial Analysis Evaluation

This script compares results from different evaluation methods (no-shot, few-shot, fine-tuned)
and generates a simple comparison report.

Usage:
    python simple_comparator.py --results-dir results --output-file comparison.json
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def compare_results(results_dir, output_file):
    """
    Compare results from different evaluation methods.

    Args:
        results_dir: Directory containing evaluation results
        output_file: Path to output file for comparison results
    """
    # Check which methods have results
    methods = []
    summaries = {}

    for method in ["no_shot", "few_shot", "fine_tuned"]:
        summary_path = os.path.join(results_dir, f"{method}_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                summaries[method] = summary
                methods.append(method)

    if not methods:
        print("No evaluation results found")
        return

    print(f"Found results for {len(methods)} methods: {', '.join(methods)}")

    # Key metrics to compare
    metrics = [
        "result_match_rate", "precision", "recall", "f1_score",
        "code_correctness", "response_time", "tokens_used"
    ]

    # Create comparison data
    comparison = {
        "methods": methods,
        "metrics": {}
    }

    # Compare metrics
    for metric in metrics:
        comparison["metrics"][metric] = {
            method: summaries[method].get(metric, 0.0) for method in methods
        }

    # Calculate improvements if multiple methods are available
    if len(methods) > 1:
        if "no_shot" in methods and "few_shot" in methods:
            comparison["improvements"] = {
                "few_shot_vs_no_shot": {}
            }
            for metric in metrics:
                no_shot_value = summaries["no_shot"].get(metric, 0.0)
                few_shot_value = summaries["few_shot"].get(metric, 0.0)
                if no_shot_value > 0:
                    improvement = (
                        (few_shot_value - no_shot_value) / no_shot_value) * 100
                    comparison["improvements"]["few_shot_vs_no_shot"][metric] = improvement
                else:
                    comparison["improvements"]["few_shot_vs_no_shot"][metric] = 0.0

        if "few_shot" in methods and "fine_tuned" in methods:
            if "improvements" not in comparison:
                comparison["improvements"] = {}
            comparison["improvements"]["fine_tuned_vs_few_shot"] = {}
            for metric in metrics:
                few_shot_value = summaries["few_shot"].get(metric, 0.0)
                fine_tuned_value = summaries["fine_tuned"].get(metric, 0.0)
                if few_shot_value > 0:
                    improvement = (
                        (fine_tuned_value - few_shot_value) / few_shot_value) * 100
                    comparison["improvements"]["fine_tuned_vs_few_shot"][metric] = improvement
                else:
                    comparison["improvements"]["fine_tuned_vs_few_shot"][metric] = 0.0

    # Save comparison results
    with open(output_file, 'w') as f:
        json.dump(comparison, indent=2, fp=f)

    print(f"Comparison saved to {output_file}")

    # Print summary
    print("\nSummary of results:")
    for metric in ["f1_score", "code_correctness"]:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for method in methods:
            value = summaries[method].get(metric, 0.0)
            print(f"  {method.replace('_', ' ').title()}: {value:.4f}")

    # Create visualizations
    create_visualizations(comparison, os.path.dirname(output_file))

    return comparison


def create_visualizations(comparison, output_dir):
    """
    Create simple visualizations of the comparison results.

    Args:
        comparison: Comparison data
        output_dir: Directory to save visualizations
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    methods = comparison["methods"]
    metrics = list(comparison["metrics"].keys())

    # 1. Bar chart for accuracy metrics
    accuracy_metrics = ["precision", "recall", "f1_score", "result_match_rate"]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    x = np.arange(len(accuracy_metrics))

    for i, method in enumerate(methods):
        values = [comparison["metrics"][metric][method]
                  for metric in accuracy_metrics]
        ax.bar(x + i*width, values, width,
               label=method.replace("_", " ").title())

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Accuracy Metrics Comparison')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in accuracy_metrics])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_metrics.png"), dpi=300)

    # 2. Bar chart for code quality and performance
    other_metrics = ["code_correctness", "response_time", "tokens_used"]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    x = np.arange(len(other_metrics))

    for i, method in enumerate(methods):
        values = [comparison["metrics"][metric][method]
                  for metric in other_metrics]
        # Normalize response time and tokens to make them comparable on the same scale
        if "response_time" in other_metrics:
            rt_idx = other_metrics.index("response_time")
            if values[rt_idx] > 0:
                # Invert and cap at 1.0
                values[rt_idx] = min(1.0, 1.0 / values[rt_idx])
        if "tokens_used" in other_metrics:
            tu_idx = other_metrics.index("tokens_used")
            if values[tu_idx] > 0:
                # Invert and normalize
                values[tu_idx] = min(1.0, 1000.0 / values[tu_idx])

        ax.bar(x + i*width, values, width,
               label=method.replace("_", " ").title())

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (normalized)')
    ax.set_title('Code Quality and Performance Metrics')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in other_metrics])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "other_metrics.png"), dpi=300)

    # 3. Improvements chart (if available)
    if "improvements" in comparison:
        for comp_name, comp_data in comparison["improvements"].items():
            metrics_to_plot = [m for m in metrics if m in comp_data]
            values = [comp_data[m] for m in metrics_to_plot]

            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['g' if v >= 0 else 'r' for v in values]
            ax.bar(metrics_to_plot, values, color=colors)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Metric')
            ax.set_ylabel('Improvement (%)')
            ax.set_title(f'{comp_name.replace("_", " ").title()} Improvement')
            ax.set_xticklabels([m.replace("_", " ").title()
                               for m in metrics_to_plot], rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig(os.path.join(
                output_dir, f"{comp_name}_improvement.png"), dpi=300)

    print(f"Visualizations saved to {output_dir}")


def main():
    """Parse arguments and run comparison."""
    parser = argparse.ArgumentParser(
        description="Compare spatial analysis evaluation results")
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing evaluation results")
    parser.add_argument("--output-file", default="comparison.json",
                        help="Output file for comparison results")
    args = parser.parse_args()

    # Make sure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run comparison
    compare_results(args.results_dir, args.output_file)


if __name__ == "__main__":
    main()
