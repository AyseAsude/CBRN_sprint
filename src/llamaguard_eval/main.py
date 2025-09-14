import argparse
import json
import pandas as pd
from datasets import load_dataset as hf_load

from src.llamaguard_eval.authenticate_hf import authenticate
from models import load_llama_guard, classify_instruction_answer
from parser import parse_output
from utils import save_results, normalize_dataset


def load_dataset(path: str, file_type: str = "csv", split: str = "train"):
    """
    Load dataset from CSV, JSON, or Hugging Face hub.
    """
    if file_type == "csv":
        df = pd.read_csv(path)
        dataset = df.to_dict(orient="records")

    elif file_type == "json":
        with open(path, "r") as f:
            dataset = json.load(f)

    elif file_type == "hf":
        ds = hf_load(path, split=split)
        dataset = ds.to_list()

    else:
        raise ValueError("file_type must be one of: 'csv', 'json', 'hf'")

    return normalize_dataset(dataset)


def run_evaluation(models, dataset, output_prefix="results", max_examples=None):
    all_results = []

    for model_name in models:
        print(f"🔍 Evaluating with {model_name} ...")
        tokenizer, model = load_llama_guard(model_name)

        for idx, row in enumerate(dataset):
            if max_examples and idx >= max_examples:
                break

            instruction = row["instruction"]
            answer = row["answer"]

            raw_output = classify_instruction_answer(tokenizer, model, instruction, answer)
            is_safe = parse_output(raw_output)

            result = {
                "example_id": idx,
                "model_name": model_name,
                "instruction": instruction,
                "answer": answer,
                "raw_output": raw_output,
                "is_safe": is_safe
            }
            all_results.append(result)

    save_results(all_results, output_prefix=output_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["meta-llama/Llama-Guard-3-8B"])
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to CSV/JSON file or Hugging Face dataset name")
    parser.add_argument("--dataset_type", choices=["csv", "json", "hf"], default="csv")
    parser.add_argument("--split", type=str, default="train",
                        help="Split to use if loading from Hugging Face")
    parser.add_argument("--output_prefix", type=str, default="llamaguard_eval")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples for quick testing")

    args = parser.parse_args()

    # 🔑 Authenticate Hugging Face first
    authenticate()

    dataset = load_dataset(args.dataset_path, file_type=args.dataset_type, split=args.split)
    run_evaluation(args.models, dataset, output_prefix=args.output_prefix, max_examples=args.max_examples)
