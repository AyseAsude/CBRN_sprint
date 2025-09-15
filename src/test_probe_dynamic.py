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
    """Dataset for dynamic generation - only questions provided, answers generated."""

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

            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Additional cleanup: remove common assistant prefixes if they exist
            prefixes_to_remove = ['assistant\n\n', 'assistant\n', 'assistant:', 'assistant ', 'Assistant\n\n', 'Assistant\n', 'Assistant:', 'Assistant ']
            for prefix in prefixes_to_remove:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
                    break
            
            dataset_source = item.get('dataset_source', 'unknown')

            # Store generated data
            self.generated_data.append({
                'instruction': question,
                'output': answer,
                'dataset_source': dataset_source
            })

        logger.info(f"Answer generation completed for {len(self.generated_data)} samples")

    def __len__(self):
        return len(self.generated_data)

    def __getitem__(self, idx):
        item = self.generated_data[idx]

        # Format as conversation
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

        # Tokenize
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
            "text": text
        }


def load_dynamic_test_data(test_data_config, generation_config, tokenizer, base_model, batch_size=16, max_length=2048):
    """Load test dataset and generate answers dynamically."""
    import random

    all_paths = test_data_config['paths']

    all_data = []
    for path in all_paths:
        logger.info(f"Loading test questions from {path}")
        data = load_dataset(path)
        
        # Add dataset source to each item
        dataset_name = os.path.basename(path)  # or use the full path if you prefer
        for item in data:
            item['dataset_source'] = dataset_name
        
        all_data.extend(data)
        logger.info(f"  Loaded {len(data)} samples from {path}")

    # Combine and shuffle
    random.shuffle(all_data)

    logger.info(f"Test set: {len(all_data)} questions")

    # Create dynamic dataset (generates answers)
    if 'temperature' not in generation_config:
        logger.error("temperature not found in generation config")
        raise ValueError("temperature missing from generation config")
    if 'max_new_tokens' not in generation_config:
        logger.error("max_new_tokens not found in generation config")
        raise ValueError("max_new_tokens missing from generation config")

    test_dataset = DynamicProbeDataset(
        all_data,
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
    """Main function for dynamic generation."""
    logger.info("Starting Dynamic generation")

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


    # Create model for generation
    model, tokenizer = create_probe_model(config)

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

    logger.info("Generation completed")

    # Save generated outputs
    output_dir = test_config.get('checkpoint_dir')
    os.makedirs(output_dir, exist_ok=True)

    outputs_file = os.path.join(output_dir, f"generated_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    output_data = {
        'generation_params': {
            'temperature': generation_config['temperature'],
            'max_new_tokens': generation_config['max_new_tokens']
        },
        'config': test_config,
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(generated_data),
        'outputs': generated_data
    }

    with open(outputs_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Generated outputs saved to {outputs_file}")

    return output_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Text Generation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (overrides checkpoint_dir from config)")

    args = parser.parse_args()

    test_probe_dynamic(args.config, checkpoint_override=args.checkpoint)