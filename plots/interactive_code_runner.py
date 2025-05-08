import json
import os
import subprocess
import tempfile


def interactive_code_runner():
    """Interactive tool to run code from JSON files."""
    # Get list of JSON files in current directory
    json_files = [f for f in os.listdir(
        './data') if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in current directory.")
        return

    # Show list of JSON files
    print("Available JSON files:")
    for i, file in enumerate(json_files):
        print(f"{i+1}. {file}")

    # Choose a file
    while True:
        try:
            file_idx = int(input("\nEnter file number: ")) - 1
            if 0 <= file_idx < len(json_files):
                break
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")

    file_path = os.path.join('./data', json_files[file_idx])

    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        samples = json.load(f)

    # Run all code samples without prompting
    import tempfile
    import subprocess

    success_count = 0
    failure_count = 0
    samples_with_code = [s for s in samples if s.get("Code")]
    total = len(samples_with_code)

    for idx, sample in enumerate(samples_with_code):
        query = sample.get("Query", "Unknown query")
        code = sample["Code"]
        print(f"\n=== Running sample {idx+1}/{total}: {query} ===")
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ['python', tmp_path],
                cwd=os.path.dirname(file_path),     # <— this will be './data'
                capture_output=True,
                text=True
            )

            # Show output and errors
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("\nErrors:")
                print(result.stderr)
            # Count success or failure
            if result.returncode == 0:
                success_count += 1
            else:
                failure_count += 1
        finally:
            os.remove(tmp_path)

    print(
        f"\nCompleted execution: {success_count}/{total} succeeded, {failure_count} failed.")


# Run the interactive tool
interactive_code_runner()
