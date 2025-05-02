#!/usr/bin/env python
"""
Fine-Tuning Dataset Preparation

This script converts formatted spatial analysis samples into a JSONL file
suitable for fine-tuning with OpenAI.

Usage:
    python prepare_fine_tuning.py --samples-file samples.json --output-file fine_tuning.jsonl
"""

import json
import argparse
from typing import List, Dict, Any


def create_fine_tuning_dataset(samples_file: str, output_file: str) -> None:
    """
    Create a fine-tuning dataset in JSONL format from spatial analysis samples.

    Args:
        samples_file: Path to JSON file with formatted samples
        output_file: Path to output JSONL file for fine-tuning
    """
    # Load samples
    with open(samples_file, 'r') as f:
        samples = json.load(f)

    # Create fine-tuning data in the required format
    fine_tuning_data = []

    for sample in samples:
        # Skip samples without both Query and Code
        if not sample.get("Query") or not sample.get("Code"):
            continue

        # Create message format required by OpenAI
        messages = [
            {"role": "system", "content": "You are a spatial analysis assistant that generates Python code using GeoPandas for GIS queries."},
            {"role": "user",
                "content": f"Generate Python code for this spatial analysis query: {sample['Query']}"},
            {"role": "assistant", "content": sample['Code']}
        ]

        # Add to dataset
        fine_tuning_data.append({"messages": messages})

    # Write to JSONL file (one JSON object per line)
    with open(output_file, 'w') as f:
        for item in fine_tuning_data:
            f.write(json.dumps(item) + '\n')

    print(
        f"Created fine-tuning dataset with {len(fine_tuning_data)} examples at {output_file}")


def validate_dataset(jsonl_file: str) -> bool:
    """
    Validate that the fine-tuning dataset meets OpenAI's requirements.

    Args:
        jsonl_file: Path to JSONL file

    Returns:
        True if valid, False otherwise
    """
    try:
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            # Check each line is valid JSON
            data = json.loads(line)

            # Check for 'messages' field
            if 'messages' not in data:
                print(f"Error on line {i+1}: Missing 'messages' field")
                return False

            # Check message format
            messages = data['messages']
            if not isinstance(messages, list) or len(messages) < 1:
                print(
                    f"Error on line {i+1}: 'messages' must be a non-empty array")
                return False

            # Check roles and content
            for j, message in enumerate(messages):
                if not isinstance(message, dict):
                    print(
                        f"Error on line {i+1}, message {j+1}: Message must be an object")
                    return False

                if 'role' not in message:
                    print(
                        f"Error on line {i+1}, message {j+1}: Missing 'role' field")
                    return False

                if message['role'] not in ['system', 'user', 'assistant']:
                    print(
                        f"Error on line {i+1}, message {j+1}: Invalid role '{message['role']}'")
                    return False

                if 'content' not in message:
                    print(
                        f"Error on line {i+1}, message {j+1}: Missing 'content' field")
                    return False

                if not isinstance(message['content'], str):
                    print(
                        f"Error on line {i+1}, message {j+1}: 'content' must be a string")
                    return False

                if not message['content'].strip():
                    print(
                        f"Error on line {i+1}, message {j+1}: 'content' must not be empty")
                    return False

        print(
            f"Dataset validation successful: {len(lines)} examples are valid")
        return True

    except Exception as e:
        print(f"Error validating dataset: {e}")
        return False


def estimate_fine_tuning_cost(jsonl_file: str) -> None:
    """
    Estimate the cost of fine-tuning based on token count.

    Args:
        jsonl_file: Path to JSONL file
    """
    try:
        # This is a very rough estimation - actual costs will vary
        with open(jsonl_file, 'r') as f:
            content = f.read()

        # Rough estimate: 1 token ≈ 4 characters for English text
        num_chars = len(content)
        estimated_tokens = num_chars / 4

        # Current pricing (as of 2023) is around $0.008 per 1K tokens for GPT-3.5
        # and around $0.12 per 1K tokens for GPT-4
        estimated_cost_gpt35 = estimated_tokens / 1000 * 0.008
        estimated_cost_gpt4 = estimated_tokens / 1000 * 0.12

        print(f"Estimated fine-tuning costs:")
        print(f"  - Approximate token count: {int(estimated_tokens)}")
        print(f"  - Estimated cost (GPT-3.5): ${estimated_cost_gpt35:.2f}")
        print(f"  - Estimated cost (GPT-4): ${estimated_cost_gpt4:.2f}")
        print("Note: These are rough estimates. Actual costs may vary based on OpenAI pricing.")

    except Exception as e:
        print(f"Error estimating cost: {e}")


def main():
    """Parse arguments and prepare fine-tuning dataset."""
    parser = argparse.ArgumentParser(
        description="Prepare fine-tuning dataset from spatial analysis samples")
    parser.add_argument("--samples-file", required=True,
                        help="Path to samples JSON file")
    parser.add_argument("--output-file", required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--validate", action="store_true",
                        help="Validate dataset after creation")
    parser.add_argument("--estimate-cost", action="store_true",
                        help="Estimate fine-tuning cost")
    args = parser.parse_args()

    # Create dataset
    create_fine_tuning_dataset(args.samples_file, args.output_file)

    # Validate if requested
    if args.validate:
        validate_dataset(args.output_file)

    # Estimate cost if requested
    if args.estimate_cost:
        estimate_fine_tuning_cost(args.output_file)


if __name__ == "__main__":
    main()
