import json
from datasets import Dataset
from tqdm import tqdm

def create_prompt(diff):
    return f"""Given the following git diff, generate a conventional commit message.
### Git Diff:
{diff}
### Commit Message:
"""

def prepare_data(raw_data_path="data/raw_data.jsonl", processed_data_path="data/processed_data.jsonl"):
    with open(raw_data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    with open(processed_data_path, "w") as f:
        for item in tqdm(raw_data, desc="Processing data"):
            prompt = create_prompt(item["diff"])
            processed_item = {"text": prompt + item["commit_message"]}
            f.write(json.dumps(processed_item) + "\n")

    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    prepare_data()
