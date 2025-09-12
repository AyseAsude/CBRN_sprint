"""
Data utilities for loading and preprocessing calibration datasets.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class ProbeDataset(Dataset):
    """Dataset for probe training with (instruction, llm answer, harmfulness) triplets."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as conversation for instruct models
        messages = [
            {"role": "user", "content": item['instruction']},
            {"role": "assistant", "content": item['output']}
        ]
        
        # Apply chat template if available, otherwise fall back to simple format
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            # Raise error if model doesn't have chat template
            raise ValueError(f"Model {self.tokenizer.name_or_path} does not have a chat template. "
                           "Chat templates are required for proper formatting of instruct models.")
        
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
            "labels": torch.tensor(int(item["harmful"]), dtype=torch.long),
            "text": text
        }


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples")
    return data


def load_combined_dataset(
    benign_path: str,
    harmful_path: str,
    val_split: float = 0.2,
    random_state: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load and combine datasets from two JSON files with labels, then perform stratified split.
    
    Args:
        benign_path: Path to benign samples JSON (label=0)
        harmful_path: Path to harmful samples JSON (label=1)
        val_split: Validation split ratio (default 0.2 for 80/20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data)
    """
    logger.info(f"Loading benign samples from {benign_path}")
    with open(benign_path, 'r') as f:
        benign_data = json.load(f)
    
    logger.info(f"Loading harmful samples from {harmful_path}")
    with open(harmful_path, 'r') as f:
        harmful_data = json.load(f)
    
    # Add harmful label to each sample
    for sample in benign_data:
        sample['harmful'] = 0
    
    for sample in harmful_data:
        sample['harmful'] = 1
    
    # Combine all data
    all_data = benign_data + harmful_data
    
    logger.info(f"Total samples: {len(all_data)} (benign: {len(benign_data)}, harmful: {len(harmful_data)})")
    
    # Extract labels for stratified split
    labels = [sample['harmful'] for sample in all_data]
    
    # Perform stratified train/validation split
    train_data, val_data, train_labels, val_labels = train_test_split(
        all_data, 
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=random_state
    )
    
    # Log split statistics
    train_harmful_count = sum(train_labels)
    val_harmful_count = sum(val_labels)
    logger.info(f"Train set: {len(train_data)} samples (harmful: {train_harmful_count}, benign: {len(train_data) - train_harmful_count})")
    logger.info(f"Val set: {len(val_data)} samples (harmful: {val_harmful_count}, benign: {len(val_data) - val_harmful_count})")
    
    return train_data, val_data


def create_dataloaders(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]], 
    tokenizer,
    batch_size: int = 16,
    max_length: int = 2048,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, val, and test sets."""
    
    train_dataset = ProbeDataset(train_data, tokenizer, max_length)
    val_dataset = ProbeDataset(val_data, tokenizer, max_length)
    

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def load_config_and_data(config_path: str):
    """Helper function to load config and prepare data."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    train_data, val_data = load_combined_dataset(benign_path=config['data']['benign_path'],
                                                harmful_path=config['data']['harmful_path'],
                                                val_split=config['data']['val_split'],
                                                random_state=config['seed'])


    train_loader, val_loader = create_dataloaders(
        train_data, val_data,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_length']
    )
    
    return config, tokenizer, train_loader, val_loader