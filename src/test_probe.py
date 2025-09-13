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


def load_test_data(test_data_config: dict, tokenizer, batch_size: int = 16, max_length: int = 2048, use_output: bool = True):
    """Load test dataset from separate benign and harmful files and create dataloader."""
    import random

    benign_paths = test_data_config['benign']
    harmful_paths = test_data_config['harmful']

    # Load all benign data
    benign_data = []
    for path in benign_paths:
        logger.info(f"Loading benign test data from {path}")
        data = load_dataset(path)
        benign_data.extend(data)
        logger.info(f"  Loaded {len(data)} samples from {path}")

    # Load all harmful data
    harmful_data = []
    for path in harmful_paths:
        logger.info(f"Loading harmful test data from {path}")
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
    
    logger.info(f"Test set: {len(test_data)} samples (benign: {len(benign_data)}, harmful: {len(harmful_data)})")
    
    test_dataset = ProbeDataset(test_data, tokenizer, max_length, use_output=use_output)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Already shuffled
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader, test_data



def test_probe(config_path: str):
    """Main testing function for probe model."""
    logger.info("Starting Probe model testing")
    
    with open(config_path, 'r') as f:
        test_config = yaml.safe_load(f)
    
    if 'test_data' not in test_config:
        logger.error("test_data not found in config. Please add it to your config file.")
        raise ValueError("test_data missing from config")
    
    if 'checkpoint_dir' not in test_config:
        logger.error("checkpoint_dir not found in config. Please add it to your config file.")
        raise ValueError("checkpoint_dir missing from config")

    test_data_config = test_config['test_data']
    checkpoint_dir = test_config['checkpoint_dir']
    
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Model checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    training_config_path = os.path.join(checkpoint_dir, "config.yaml")
    
    if os.path.exists(training_config_path):
        logger.info(f"Loading training config from {training_config_path}")
        with open(training_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
    else:
        logger.error(f"Training config not found at {training_config_path}")
        raise FileNotFoundError(f"Training config not found at {training_config_path}")

    
    model, tokenizer = create_probe_model(config)
    
    logger.info(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    base_model_device = next(model.base_model.parameters()).device
    model.classifier = model.classifier.to(base_model_device)
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Check if use_output is specified in config, default to True for backward compatibility
    use_output = config['data'].get('use_output', True)

    test_loader, _ = load_test_data(
        test_data_config=test_data_config,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_length'],
        use_output=use_output
    )
    
    logger.info("Starting evaluation on test set")
    test_metrics = evaluate_model(model, test_loader, detailed=True)
    
    print_test_metrics(test_metrics)
    
    test_results = {
        'config': config,
        'test_metrics': test_metrics,
        'checkpoint_metrics': checkpoint.get('val_metrics', {}),
        'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'test_timestamp': datetime.now().isoformat()
    }
    
    results_file = os.path.join(checkpoint_dir, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Test results saved to {results_file}")
    
    predictions_file = os.path.join(checkpoint_dir, "test_predictions.json")
    predictions_data = []
    for i in range(len(test_metrics['labels'])):
        predictions_data.append({
            'index': i,
            'true_label': test_metrics['labels'][i],
            'predicted_label': test_metrics['predictions'][i],
            'harmful_probability': test_metrics['probabilities'][i]
        })
    
    with open(predictions_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    logger.info(f"Detailed predictions saved to {predictions_file}")
    
    return test_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Probe Model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    test_probe(args.config)