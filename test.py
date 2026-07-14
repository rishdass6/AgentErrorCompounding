def pull_human_eval():
    """
    Pulls task 32 from HumanEval-XL and prints the prompt, ground-truth
    solution, and tests.

    Why not `datasets.load_dataset("floatai/HumanEval-XL", ...)`:
    that repo uses a loading script (HumanEval-XL.py), and newer versions
    of the `datasets` library (>=3.0) dropped support for script-based
    datasets entirely, so it fails with "Dataset scripts are no longer
    supported" regardless of trust_remote_code. Easiest fix is to just
    pull the raw JSONL straight from the GitHub repo instead.

    pip install requests
    """

    import json
    import requests

    PROGRAMMING_LANGUAGE = "python"   # python/java/go/csharp/kotlin/perl/php/ruby/scala/swift/typescript/javascript
    NATURAL_LANGUAGE = "English"      # English/Chinese/Russian/German/... (23 total)
    TASK_IDS = ["python/49", "python/59", "python/62", "python/63", "python/72", "python/74", "python/60", "python/78"]

    url = (
        "https://raw.githubusercontent.com/floatai/HumanEval-XL/main/data/"
        f"{PROGRAMMING_LANGUAGE}/{NATURAL_LANGUAGE}.jsonl"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    tasks = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    for task in (t for t in tasks if t["task_id"] in TASK_IDS):

        print("=" * 20, "TASK ID", "=" * 20)
        print(task["task_id"])

        print("\n" + "=" * 20, "PROMPT / INSTRUCTION", "=" * 20)
        print(task["prompt"])

        print("\n" + "=" * 20, "GROUND TRUTH SOLUTION", "=" * 20)
        print(task["canonical_solution"])

        print("\n" + "=" * 20, "TESTS", "=" * 20)
        print(task["test"])

        print("\n" + "=" * 20, "ENTRY POINT", "=" * 20)
        print(task["entry_point"])

from datasets import load_dataset

def fetch_hardmath_example():
    print("--- FETCHING HARDMath DATASET ---")
    try:
        # Load the default split for HARDMath2 from Hugging Face
        dataset = load_dataset("JVRoggeveen/HARDMath2", split="train")
        
        # Take the first entry in the dataset
        first_entry = dataset[0]
        
        # Parse based on HARDMath's native dictionary schema ('prompt' & 'solution')
        prompt = first_entry.get("prompt", "No prompt found")
        solution = first_entry.get("solution", "No solution found")
        
        print(f"PROMPT:\n{prompt}\n")
        print(f"SOLUTION:\n{solution}\n")
    except Exception as e:
        print(f"Error loading HARDMath: {e}\n")

from datasets import load_dataset

def get_musr_example():
    print("--- FETCHING MuSR DATASET EXAMPLE ---")
    try:
        # FIX: Explicitly target the 'default' configuration, and load 'murder_mysteries' as the split
        dataset = load_dataset("TAUR-Lab/MuSR", "default", split="murder_mysteries")
        
        # Pull the first puzzle row
        example = dataset[0]
        
        # Extract individual keys from the dataset schema
        narrative = example.get("narrative", "")
        question = example.get("question", "")
        choices = example.get("choices", [])
        answer_index = example.get("answer_index", 0)
        
        # Format the full prompt exactly how it is fed to frontier models
        full_prompt = (
            f"--- CONTEXT NARRATIVE ---\n{narrative}\n"
            f"--------------------------\n"
            f"QUESTION: {question}\n\n"
            f"CHOICES:\n"
        )
        for i, choice in enumerate(choices):
            full_prompt += f" [{i}] {choice}\n"
            
        # Extract the correct string answer using the ground truth index
        ground_truth_text = choices[answer_index] if choices else "Unknown"
        
        # Output results
        print(full_prompt)
        print(f"--- GROUND TRUTH ---")
        print(f"Correct Index: {answer_index}")
        print(f"Correct Text Answer: {ground_truth_text}\n")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_musr_example()
