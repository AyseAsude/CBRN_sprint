import os
import yaml
import torch
from torch.utils.data import DataLoader
import logging
import json
from datetime import datetime
from data_utils import ProbeDataset, load_dataset
from models import create_probe_model
from evaluation import evaluate_model, print_test_metrics
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import time

log_filename = f"testing_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


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


def get_probe_probability(model, tokenizer, instruction, answer, max_length=2048, use_output=True):
    """Get probe probability for a single instruction+answer pair."""
    messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": answer}
                ]

    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

    # Tokenize the text
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to model device
    device = next(model.parameters()).device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs['logits'], dim=-1)
        harmful_probability = probabilities[0, 1].item()  # Probability of class 1 (harmful)
    
    return harmful_probability


def process_json_with_probe(config_path: str):
    """Process JSON file and add probe probabilities to each example."""
    logger.info("Starting JSON processing with Probe model")
    
    with open(config_path, 'r') as f:
        test_config = yaml.safe_load(f)
    
    if 'checkpoint_dir' not in test_config:
        logger.error("checkpoint_dir not found in config. Please add it to your config file.")
        raise ValueError("checkpoint_dir missing from config")

    checkpoint_dir = test_config['checkpoint_dir']
    
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    training_config_path = os.path.join(checkpoint_dir, "config.yaml")
    
    logger.info(f"Loading training config from {training_config_path}")
    with open(training_config_path, 'r') as f:
        config = yaml.safe_load(f)
    

    model, tokenizer = create_probe_model(config)
    
    logger.info(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    base_model_device = next(model.base_model.parameters()).device
    model.classifier = model.classifier.to(base_model_device)
    
    logger.info(f"Model loaded successfully")
    
    input_json_path = test_config['input_json_path']
    output_json_path = test_config['output_json_path'
    ]

    logger.info(f"Loading JSON data from {input_json_path}")
    with open(input_json_path, 'r') as f:
        examples = json.load(f)
    
    logger.info(f"Processing {len(examples)} examples")
    
    max_length = config['model']['max_length']
    
    true_labels = []
    predicted_probs = []

    processing_start_time = time.time()

    for i, example in enumerate(examples):
        instruction = example['instruction']
        answer = example['answer']
        
        is_safe_value = example['is_safe']
        if is_safe_value is True:
            true_label = 0  # safe
        elif is_safe_value is False:
            true_label = 1  # unsafe
        else:
            raise ValueError(f"Invalid is_safe value: {is_safe_value}")
        
        true_labels.append(true_label)

        harmful_probability = get_probe_probability(
            model, tokenizer, instruction, answer, max_length
        )

        predicted_probs.append(harmful_probability)
        example['probe_harmful_probability'] = harmful_probability
        
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(examples)} examples")
    
    processing_end_time = time.time()
    processing_time = processing_end_time - processing_start_time

    avg_time_per_example = processing_time / len(examples)

    logger.info(f"=== PERFORMANCE METRICS ===")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"Average time per example: {avg_time_per_example:.4f} seconds")


    logger.info(f"Saving results to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(examples, f, indent=2)


    predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probs]
    
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    metrics_results = {
        'f1_score': f1,
        'accuracy': accuracy,
        'num_examples': len(examples),
        'num_safe': sum(1 for label in true_labels if label == 0),
        'num_unsafe': sum(1 for label in true_labels if label == 1),
        'threshold': 0.5,
        'processing_time': processing_time,
        'avg_time_per_ex': avg_time_per_example,
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_file = output_json_path.replace('.json', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_results, f, indent=2)
    
    logger.info(f"Metrics results saved to {metrics_file}")
    
    logger.info("Processing complete")
    
    return examples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Probe Model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    process_json_with_probe(args.config)