import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
import logging
import json
from datetime import datetime
from tqdm.auto import tqdm
from data_utils import load_dataset
from models import create_probe_model
from evaluation import evaluate_model, print_test_metrics


log_filename = f"testing_probe_dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


logging.getLogger().handlers.clear()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class DynamicProbeDataset(Dataset):
    """Dataset for dynamic probe evaluation - only questions provided, answers generated."""

    def __init__(self, data, tokenizer, base_model, max_length=2048, temperature=0.7, max_new_tokens=1024):
        self.questions = data
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.max_length = max_length
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.generated_data = []

        logger.info(f"Generating answers for {len(self.questions)} questions...")
        self._generate_answers()

    def _generate_answers(self):
        """Generate answers using the base model for all questions."""
        self.base_model.eval()

        for item in tqdm(self.questions, desc="Generating answers"):
            question = item['instruction']

            # Format question for generation
            messages = [
                {"role": "user", "content": question}
            ]

            # Apply chat template
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                raise ValueError(f"Model {self.tokenizer.name_or_path} does not have a chat template.")

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )

            # Move to model device
            device = next(self.base_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate answer
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode answer
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if prompt in generated_text:
                answer = generated_text[len(prompt):].strip()
            else:
                # Handle case where prompt might have been modified during generation
                answer = generated_text.split(question)[-1].strip()

            # Store generated data
            self.generated_data.append({
                'instruction': question,
                'output': answer,
                'harmful': item.get('harmful', 0)  # Keep original label if exists
            })

        logger.info(f"Answer generation completed for {len(self.generated_data)} samples")

    def __len__(self):
        return len(self.generated_data)

    def __getitem__(self, idx):
        item = self.generated_data[idx]

        # Format as conversation for probe evaluation
        messages = [
            {"role": "user", "content": item['instruction']},
            {"role": "assistant", "content": item['output']}
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize for probe
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(item["harmful"]), dtype=torch.long),
            "text": text
        }


def load_dynamic_test_data(test_data_config, generation_config, tokenizer, base_model, batch_size=16, max_length=2048):
    """Load test dataset and generate answers dynamically."""
    import random

    benign_paths = test_data_config['benign']
    harmful_paths = test_data_config['harmful']

    # Load all benign data
    benign_data = []
    for path in benign_paths:
        logger.info(f"Loading benign test questions from {path}")
        data = load_dataset(path)
        benign_data.extend(data)
        logger.info(f"  Loaded {len(data)} samples from {path}")

    # Load all harmful data
    harmful_data = []
    for path in harmful_paths:
        logger.info(f"Loading harmful test questions from {path}")
        data = load_dataset(path)
        harmful_data.extend(data)
        logger.info(f"  Loaded {len(data)} samples from {path}")

    # Add labels
    for sample in benign_data:
        sample['harmful'] = 0

    for sample in harmful_data:
        sample['harmful'] = 1

    # Combine and shuffle
    test_data = benign_data + harmful_data
    random.shuffle(test_data)

    logger.info(f"Test set: {len(test_data)} questions (benign: {len(benign_data)}, harmful: {len(harmful_data)})")

    # Create dynamic dataset (generates answers)
    if 'temperature' not in generation_config:
        logger.error("temperature not found in generation config")
        raise ValueError("temperature missing from generation config")
    if 'max_new_tokens' not in generation_config:
        logger.error("max_new_tokens not found in generation config")
        raise ValueError("max_new_tokens missing from generation config")

    test_dataset = DynamicProbeDataset(
        test_data,
        tokenizer,
        base_model,
        max_length=max_length,
        temperature=generation_config['temperature'],
        max_new_tokens=generation_config['max_new_tokens']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return test_loader, test_dataset.generated_data


def test_probe_dynamic(config_path: str, checkpoint_override: str = None):
    """Main testing function for dynamic probe evaluation."""
    logger.info("Starting Dynamic Probe model testing")

    with open(config_path, 'r') as f:
        test_config = yaml.safe_load(f)

    if 'test_data' not in test_config:
        logger.error("test_data not found in config. Please add it to your config file.")
        raise ValueError("test_data missing from config")

    test_data_config = test_config['test_data']

    # Determine checkpoint path
    if checkpoint_override:
        checkpoint_path = checkpoint_override
        checkpoint_dir = os.path.dirname(checkpoint_path)
        logger.info(f"Using checkpoint override: {checkpoint_path}")
        # For layer-wise checkpoints, config is in parent directory
        parent_dir = os.path.dirname(checkpoint_dir)
        training_config_path = os.path.join(parent_dir, "config.yaml")
    else:
        checkpoint_dir = test_config['checkpoint_dir']
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        training_config_path = os.path.join(checkpoint_dir, "config.yaml")

    if not os.path.exists(checkpoint_path):
        logger.error(f"Model checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    if os.path.exists(training_config_path):
        logger.info(f"Loading training config from {training_config_path}")
        with open(training_config_path, 'r') as f:
            config = yaml.safe_load(f)

    else:
        logger.error(f"Training config not found at {training_config_path}")
        raise FileNotFoundError(f"Training config not found at {training_config_path}")


    # Create model with base model for generation
    model, tokenizer = create_probe_model(config)

    logger.info(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])

    base_model_device = next(model.base_model.parameters()).device
    model.classifier = model.classifier.to(base_model_device)

    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Get generation config (will be validated in load_dynamic_test_data)
    generation_config = test_config.get('generation', {})

    # Load test data and generate answers dynamically
    test_loader, generated_data = load_dynamic_test_data(
        test_data_config=test_data_config,
        generation_config=generation_config,
        tokenizer=tokenizer,
        base_model=model.base_model,  # Pass base model for generation
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_length']
    )

    logger.info("Starting evaluation on dynamically generated test set")
    test_metrics = evaluate_model(model, test_loader, detailed=True)

    print_test_metrics(test_metrics)

    test_results = {
        'config': config,
        'test_metrics': test_metrics,
        'checkpoint_metrics': checkpoint.get('val_metrics', {}),
        'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'test_timestamp': datetime.now().isoformat(),
        'dynamic_generation': True,
        'generation_params': {
            'temperature': generation_config['temperature'],
            'max_new_tokens': generation_config['max_new_tokens']
        }
    }

    results_file = os.path.join(checkpoint_dir, "test_results_dynamic.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Test results saved to {results_file}")

    # Save predictions with generated answers
    predictions_file = os.path.join(checkpoint_dir, "test_predictions_dynamic.json")
    predictions_data = []
    for i in range(len(test_metrics['labels'])):
        predictions_data.append({
            'index': i,
            'instruction': generated_data[i]['instruction'],
            'generated_answer': generated_data[i]['output'],
            'true_label': test_metrics['labels'][i],
            'predicted_label': test_metrics['predictions'][i],
            'harmful_probability': test_metrics['probabilities'][i]
        })

    with open(predictions_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)

    logger.info(f"Detailed predictions with generated answers saved to {predictions_file}")

    return test_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Probe Model with Dynamic Generation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (overrides checkpoint_dir from config)")

    args = parser.parse_args()

    test_probe_dynamic(args.config, checkpoint_override=args.checkpoint)