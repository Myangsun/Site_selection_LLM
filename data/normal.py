import json
import ast


def make_answers_python_literals(input_path: str, output_path: str):
    """
    Reads a JSON file of samples, converts each Answer into a Python-style
    list literal string (with single quotes), and writes the result out.
    """
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        samples = json.load(f)

    for sample in samples:
        ans = sample.get("Answer")
        # If it's already a string that looks like a Python list literal, leave it
        if isinstance(ans, str) and ans.startswith('[') and ans.endswith(']'):
            # assume it’s already in the right format
            continue

        # If it's a JSON list, convert it to a Python-style literal
        if isinstance(ans, list):
            sample["Answer"] = repr(ans)
        else:
            # If it’s a JSON-encoded string of a list, parse then repr
            try:
                parsed = ast.literal_eval(ans)
                if isinstance(parsed, list):
                    sample["Answer"] = repr(parsed)
                    continue
            except Exception:
                pass
            # Fallback: wrap whatever it is in a one-element list
            sample["Answer"] = repr([ans])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_file = "formatted_samples.json"                 # your original file
    output_file = "samples_python_answers.json"
    make_answers_python_literals(input_file, output_file)
    print(f"Wrote normalized file to {output_file}")
