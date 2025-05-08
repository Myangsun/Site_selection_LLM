import json
import re
import ast
import os


def process_data(input_text):
    """Process the data to extract queries, code, and answers."""
    samples = []

    # Find all blocks that might contain queries
    blocks = re.split(r'\n\s*\n\s*\{', input_text)

    for i, block in enumerate(blocks):
        if i > 0:  # Add the opening brace back for all but the first block
            block = '{' + block

        # Try to extract query
        query_match = re.search(r'"Query":\s*"([^"]+)"', block)
        if not query_match:
            continue

        query = query_match.group(1).strip()

        # Try to extract code
        code_start = block.find('"Code": "')
        if code_start == -1:
            continue

        code_start += 9  # Length of '"Code": "'

        # Find the end of the code block
        code_end = block.find('",', code_start)
        if code_end == -1:
            # Try alternative end pattern
            code_end = block.find('",\n', code_start)
            if code_end == -1:
                continue

        code = block[code_start:code_end].strip()

        # Replace escaped newlines with actual newlines
        code = code.replace('\\n', '\n')

        # Remove 'data/' prefix from file paths
        code = code.replace("'data/cambridge_parcels.geojson'",
                            "'cambridge_parcels.geojson'")
        code = code.replace(
            "'data/cambridge_poi_processed.geojson'", "'cambridge_poi_processed.geojson'")
        code = code.replace("'data/cambridge_census_cambridge_pct.geojson'",
                            "'cambridge_census_cambridge_pct.geojson'")
        code = code.replace("'data/cambridge_spend_processed.csv'",
                            "'cambridge_spend_processed.csv'")

        # Extract answer
        answer = []
        answer_match = re.search(r'"Answer":\s*(\[.*?\])', block, re.DOTALL)
        if answer_match:
            try:
                answer_text = answer_match.group(1)
                answer = ast.literal_eval(answer_text)
            except:
                # Try to extract as a list of strings
                try:
                    answer_strings = re.findall(r"'([^']*)'", answer_text)
                    answer = answer_strings
                except:
                    answer = []
        else:
            # Try to find answer after "Answer:" pattern
            answer_match = re.search(r'Answer:\s*(\[.*?\])', block, re.DOTALL)
            if answer_match:
                try:
                    answer_text = answer_match.group(1)
                    answer = ast.literal_eval(answer_text)
                except:
                    try:
                        answer_strings = re.findall(r"'([^']*)'", answer_text)
                        answer = answer_strings
                    except:
                        answer = []

        samples.append({
            "Query": query,
            "Code": code,
            "Answer": answer
        })

    return samples


def format_samples(samples):
    """Format samples to match the structure in spatial_samples.json."""
    formatted_samples = []

    for sample in samples:
        query = sample.get("Query", "")
        code = sample.get("Code", "")
        answer = sample.get("Answer", [])

        # Format the answer to match the sample format
        if isinstance(answer, str) and answer.startswith('[') and answer.endswith(']'):
            answer_str = answer
        elif isinstance(answer, list):
            answer_str = repr(answer)
        else:
            answer_str = repr([answer])

        formatted_sample = {
            "Query": query,
            "Code": code,
            "Answer": answer_str
        }
        formatted_samples.append(formatted_sample)

    return formatted_samples


def main():
    # Read the input file
    input_path = 'paste.txt'
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found.")
        return

    with open(input_path, 'r') as file:
        input_text = file.read()

    # Process the data
    samples = process_data(input_text)

    # Format the samples
    formatted_samples = format_samples(samples)

    # Print some verification for each sample
    for i, sample in enumerate(formatted_samples):
        print(f"Sample {i+1}:")
        print(f"  Query: {sample['Query'][:50]}...")
        code_lines = sample['Code'].count('\n') + 1
        print(f"  Code: {code_lines} lines")
        print(f"  Answer: {sample['Answer'][:50]}...")
        print()

    # Save the formatted data
    output_path = 'formatted_samples.json'
    with open(output_path, 'w') as file:
        json.dump(formatted_samples, file, indent=2)

    print(
        f"Processed {len(formatted_samples)} samples and saved to {output_path}")


if __name__ == "__main__":
    main()
