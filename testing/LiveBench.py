from datasets import load_dataset

# Load the LiveBench coding dataset
# Split usually includes 'test' or 'public'
dataset = load_dataset("livebench/coding", split="test")

# Select a specific problem (e.g., the first one)
problem = dataset[3]
print(len(dataset))

# Extract the problem description to give to your Agent
print(problem["original_json"])

import sys
import json
sys.path.append("C:\Projects\Research\livebench\livebench\livebench")

from livebench.livebench.livebench.code_runner import CodeRunner

runner = CodeRunner(docker_mode=True)