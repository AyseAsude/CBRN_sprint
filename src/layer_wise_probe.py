import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd

from data_utils import load_config_and_data
from models import MLPClassifier, load_base_model
from evaluation import evaluate_model


log_filename = f"layer_wise_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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


class LayerProbeModel(nn.Module):
    """Probe model that can extract features from any specified layer."""

    def __init__(self, base_model, classifier: MLPClassifier, layer_idx: int = -1):
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.layer_idx = layer_idx

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get total number of layers
        device = next(self.base_model.parameters()).device
        sample_output = self.base_model(
            torch.zeros(1, 1, dtype=torch.long).to(device),
            output_hidden_states=True
        )
        self.num_layers = len(sample_output.hidden_states)

        # Validate layer index
        if self.layer_idx >= self.num_layers or self.layer_idx < -self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range. Model has {self.num_layers} layers.")

        logger.info(f"Probe initialized for layer {self.layer_idx} (total layers: {self.num_layers})")

    def forward(self, input_ids, attention_mask, **kwargs):
        base_model_device = next(self.base_model.parameters()).device
        input_ids = input_ids.to(base_model_device)
        attention_mask = attention_mask.to(base_model_device)

        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Extract specified layer hidden states
        hidden_states = outputs.hidden_states[self.layer_idx]

        # Mean pooling over all non-padding tokens
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled_hidden = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        features = pooled_hidden

        features = features.float()
        classifier_device = next(self.classifier.parameters()).device
        if features.device != classifier_device:
            features = features.to(classifier_device)

        logits = self.classifier(features)

        return {"logits": logits, "features": features}


def train_probe_for_layer(
    config: dict,
    train_loader,
    val_loader,
    base_model,
    layer_idx: int,
    output_base_dir: str
) -> Dict:
    """Train a probe for a specific layer and return metrics."""

    logger.info(f"Training probe for layer {layer_idx}")

    # Create output directory for this layer
    output_dir = os.path.join(output_base_dir, f"layer_{layer_idx}")
    os.makedirs(output_dir, exist_ok=True)

    # Create classifier
    classifier = MLPClassifier(
        input_size=base_model.config.hidden_size,
        hidden_sizes=config['classifier']['hidden_sizes'],
        dropout_prob=config['classifier']['dropout_prob']
    )

    # Create layer-specific probe model
    model = LayerProbeModel(base_model, classifier, layer_idx=layer_idx)

    # Move classifier to same device as base model
    base_model_device = next(model.base_model.parameters()).device
    model.classifier = model.classifier.to(base_model_device)

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    total_steps = len(train_loader) * config['training']['num_epochs']
    warmup_steps = int(total_steps * config['training']['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    model.train()
    best_val_f1 = 0
    best_val_accuracy = 0
    global_step = 0
    training_history = []
    patience_counter = 0
    patience = config['training'].get('patience', 5)

    for epoch in range(config['training']['num_epochs']):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Layer {layer_idx} - Epoch {epoch + 1}")

        for batch in progress_bar:
            model.train()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].to(base_model_device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / len(train_loader)

        # Evaluate
        val_metrics = evaluate_model(model, val_loader)

        logger.info(f"Layer {layer_idx} - Epoch {epoch + 1}: "
                   f"Loss: {avg_epoch_loss:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.4f}, "
                   f"Val F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_accuracy = val_metrics['accuracy']
            patience_counter = 0

            # Save best model
            torch.save({
                'classifier_state_dict': model.classifier.state_dict(),
                'layer_idx': layer_idx,
                'val_metrics': val_metrics,
                'epoch': epoch + 1
            }, os.path.join(output_dir, "best_model.pt"))
        else:
            patience_counter += 1

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_epoch_loss,
            'val_metrics': val_metrics
        })

        if patience_counter >= patience:
            logger.info(f"Early stopping for layer {layer_idx} at epoch {epoch + 1}")
            break

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=2)

    return {
        'layer_idx': layer_idx,
        'best_val_f1': best_val_f1,
        'best_val_accuracy': best_val_accuracy,
        'num_epochs_trained': len(training_history),
        'training_history': training_history
    }


def evaluate_all_layers(config_path: str):
    """Train and evaluate probes for multiple layers."""

    logger.info("Starting layer-wise probe analysis")

    # Load config and data
    config, tokenizer, train_loader, val_loader = load_config_and_data(config_path)

    # Load base model once
    base_model, _ = load_base_model(
        config['model']['name'],
        config['model'].get('device', 'auto')
    )

    # Determine which layers to test from config or use defaults
    if 'layers' in config:
        layer_indices = config['layers']
        logger.info(f"Using layers from config: {layer_indices}")
    else:
        # Get total number of layers
        device = next(base_model.parameters()).device
        sample_output = base_model(
            torch.zeros(1, 1, dtype=torch.long).to(device),
            output_hidden_states=True
        )
        num_layers = len(sample_output.hidden_states)

        # Test every 4th layer plus the last few
        layer_indices = list(range(0, num_layers - 4, 4)) + list(range(num_layers - 4, num_layers))
        logger.info(f"Using default layers: {layer_indices}")

    # Create output directory with timestamp
    output_base_dir = f"layer_wise_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_base_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_base_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    # Train probe for each layer
    all_results = []
    for layer_idx in layer_indices:
        result = train_probe_for_layer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            base_model=base_model,
            layer_idx=layer_idx,
            output_base_dir=output_base_dir
        )
        all_results.append(result)

        # Save intermediate results
        with open(os.path.join(output_base_dir, "layer_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Create visualization
    visualize_layer_results(all_results, output_base_dir)

    logger.info(f"Layer-wise analysis complete. Results saved to {output_base_dir}")
    return all_results


def visualize_layer_results(results: List[Dict], output_dir: str):
    """Create visualizations of layer-wise probe performance."""

    # Prepare data for plotting
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Validation accuracy
    axes[0].plot(df['layer_idx'], df['best_val_accuracy'], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Validation Accuracy by Layer')
    axes[0].grid(True, alpha=0.3)

    # Validation F1
    axes[1].plot(df['layer_idx'], df['best_val_f1'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Validation F1 Score')
    axes[1].set_title('Validation F1 Score by Layer')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_performance.png'), dpi=150)
    plt.close()

    # Create summary heatmap
    metrics_data = []
    for r in results:
        metrics_data.append({
            'Layer': r['layer_idx'],
            'Val Acc': r['best_val_accuracy'],
            'Val F1': r['best_val_f1'],
            'Epochs': r['num_epochs_trained']
        })

    df_metrics = pd.DataFrame(metrics_data)
    df_metrics = df_metrics.set_index('Layer')

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_metrics.T, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
    plt.title('Layer-wise Probe Performance Heatmap')
    plt.xlabel('Layer Index')
    plt.ylabel('Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=150)
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train probes on different layers")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")

    args = parser.parse_args()

    # Run layer-wise evaluation
    results = evaluate_all_layers(config_path=args.config)

    # Print summary
    print("\n" + "="*50)
    print("LAYER-WISE PROBE ANALYSIS SUMMARY")
    print("="*50)

    for r in results:
        print(f"Layer {r['layer_idx']:2d}: "
              f"Val Acc={r['best_val_accuracy']:.4f}, "
              f"Val F1={r['best_val_f1']:.4f}, "
              f"Epochs={r['num_epochs_trained']}")