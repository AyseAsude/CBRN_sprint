import json
import pandas as pd

def normalize_dataset(dataset):
    """
    Normalize dataset field names into 'instruction' and 'answer'.
    Auto-detects variations like 'prompt', 'response', 'question', 'output', etc.
    """
    instruction_keys = ["instruction", "prompt", "input", "question", "query", "text"]
    answer_keys = ["answer", "response", "output", "completion", "label"]

    normalized = []
    for row in dataset:
        instr, ans = None, None

        for k in instruction_keys:
            if k in row and row[k] not in [None, ""]:
                instr = row[k]
                break

        for k in answer_keys:
            if k in row and row[k] not in [None, ""]:
                ans = row[k]
                break

        if instr is None or ans is None:
            keys = list(row.keys())
            if len(keys) >= 2:
                instr = instr or row[keys[0]]
                ans = ans or row[keys[1]]

        if instr is None or ans is None:
            raise ValueError(f"Could not map fields in row: {row}")

        normalized.append({"instruction": instr, "answer": ans})

    return normalized


def save_results(results, output_prefix="results"):
    """
    Save evaluation results to CSV and JSON.
    """
    df = pd.DataFrame(results)
    df.to_csv(f"{output_prefix}.csv", index=False)

    with open(f"{output_prefix}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {output_prefix}.csv and {output_prefix}.json")
