import argparse
import json
import os
import time
import pandas as pd
from datasets import load_dataset as hf_load

from src.llamaguard_eval.authenticate_hf import authenticate
from src.llamaguard_eval.models import load_llama_guard, classify_instruction_answer
from src.llamaguard_eval.parser import parse_output
from src.llamaguard_eval.utils import save_results_json, normalize_dataset


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

    dataset = dataset['outputs']
    return normalize_dataset(dataset)


def run_evaluation(models, dataset, output_path, max_examples=None):
    all_results = []
    total_inference_time = 0

    for model_name in models:
        print(f"🔍 Evaluating with {model_name} ...")
        tokenizer, model = load_llama_guard(model_name)

        for idx, row in enumerate(dataset):
            if max_examples and idx >= max_examples:
                break

            instruction = row["instruction"]
            answer = row["answer"]

            # Time the inference
            start_time = time.time()
            raw_output = classify_instruction_answer(tokenizer, model, instruction, answer)
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            is_safe = parse_output(raw_output)

            result = {
                "example_id": idx,
                "model_name": model_name,
                "instruction": instruction,
                "answer": answer,
                "raw_output": raw_output,
                "is_safe": is_safe,
                "inference_time_seconds": round(inference_time, 3)
            }

            if "dataset_source" in row:
                result["dataset"] = row["dataset_source"]
            else:
                raise ValueError("Dataset source must be included.")

            all_results.append(result)

 
    # Print summary statistics
    avg_time = total_inference_time / len(all_results) if all_results else 0
    print(f"\n📊 Inference Statistics:")
    print(f"  Total examples: {len(all_results)}")
    print(f"  Total time: {total_inference_time:.2f}s")
    print(f"  Average time per example: {avg_time:.3f}s")

    save_results_json(all_results, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["meta-llama/Llama-Guard-3-8B"])
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to CSV/JSON file or Hugging Face dataset name")
    parser.add_argument("--dataset_type", choices=["csv", "json", "hf"], default="csv")
    parser.add_argument("--split", type=str, default="train",
                        help="Split to use if loading from Hugging Face")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples for quick testing")

    args = parser.parse_args()

    # 🔑 Authenticate Hugging Face first
    authenticate()

    # Generate output path based on input file
    if args.dataset_type in ["csv", "json"]:
        # Get directory and filename from input path
        input_dir = os.path.dirname(args.dataset_path)
        input_filename = os.path.basename(args.dataset_path)
        # Remove extension and add llamaguard_ prefix
        name_without_ext = os.path.splitext(input_filename)[0]
        output_filename = f"llamaguard_{name_without_ext}.json"
        output_path = os.path.join(input_dir if input_dir else ".", output_filename)
    else:
        # For HuggingFace datasets, save in current directory
        output_path = "llamaguard_eval.json"

    dataset = load_dataset(args.dataset_path, file_type=args.dataset_type, split=args.split)
    run_evaluation(args.models, dataset, output_path=output_path, max_examples=args.max_examples)
