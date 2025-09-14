#!/usr/bin/env python
"""
Wrapper script to evaluate all layer checkpoints using test_probe_dynamic.py
"""

import os
import json
import yaml
import subprocess
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging
log_filename = f"evaluate_layers_wrapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def run_test_probe_dynamic_for_layer(
    checkpoint_path: str,
    config_path: str,
    output_dir: str
):
    """Run test_probe_dynamic.py for a single layer checkpoint."""

    # Build command
    cmd = [
        "python", "src/test_probe_dynamic.py",
        "--config", config_path,
        "--checkpoint", checkpoint_path,  # Pass the specific layer checkpoint
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Error running test_probe_dynamic: {result.stderr}")
        return None

    # Parse the output to extract metrics
    output_lines = result.stdout.split('\n')
    metrics = {}

    for line in output_lines:
        if "Test Accuracy:" in line:
            metrics['accuracy'] = float(line.split(':')[1].strip())
        elif "Test F1:" in line:
            metrics['f1'] = float(line.split(':')[1].strip())
        elif "Test Precision:" in line:
            metrics['precision'] = float(line.split(':')[1].strip())
        elif "Test Recall:" in line:
            metrics['recall'] = float(line.split(':')[1].strip())

    # Also try to read from the metrics file if saved
    metrics_file = os.path.join(output_dir, "test_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

    return metrics


def evaluate_all_layers(
    test_benign_paths: list,
    test_harmful_paths: list,
    config_path: str = None,
):
    """Evaluate all layer checkpoints using test_probe_dynamic."""

    # Load test config
    with open(config_path, 'r') as f:
        test_config = yaml.safe_load(f)

    # Get checkpoint directory from test config
    checkpoint_dir = test_config['checkpoint_dir']

    logger.info(f"Starting evaluation of all layers from: {checkpoint_dir}")

    # Load training config to get layer indices
    training_config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    layer_indices = training_config['layers']

    logger.info(f"Will evaluate layers: {layer_indices}")

    # Create output directory
    output_base_dir = f"layer_dynamic_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_base_dir, exist_ok=True)

    # Save configs
    with open(os.path.join(output_base_dir, "test_config.yaml"), 'w') as f:
        yaml.dump(test_config, f)
    with open(os.path.join(output_base_dir, "training_config.yaml"), 'w') as f:
        yaml.dump(training_config, f)

    # Evaluate each layer
    all_results = []

    for layer_idx in layer_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Layer {layer_idx}")
        logger.info(f"{'='*60}")

        # Checkpoint path for this layer
        checkpoint_path = os.path.join(checkpoint_dir, f"layer_{layer_idx}", "best_model.pt")

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found for layer {layer_idx}")
            continue

        # Output directory for this layer
        layer_output_dir = os.path.join(output_base_dir, f"layer_{layer_idx}")
        os.makedirs(layer_output_dir, exist_ok=True)

        # Run test_probe_dynamic
        metrics = run_test_probe_dynamic_for_layer(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            output_dir=layer_output_dir
        )

        if metrics:
            metrics['layer_idx'] = layer_idx
            all_results.append(metrics)

            logger.info(f"Layer {layer_idx} Results:")
            logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"  F1: {metrics.get('f1', 'N/A'):.4f}")
            logger.info(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
            logger.info(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")

    # Save all results
    with open(os.path.join(output_base_dir, "all_layer_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create visualizations
    visualize_results(all_results, output_base_dir)

    # Print summary
    print("\n" + "="*60)
    print("LAYER-WISE DYNAMIC EVALUATION SUMMARY")
    print("="*60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_base_dir}")
    print("-"*60)

    for r in all_results:
        print(f"Layer {r['layer_idx']:2d}: "
              f"Acc={r.get('accuracy', 0):.4f}, "
              f"F1={r.get('f1', 0):.4f}, "
              f"Prec={r.get('precision', 0):.4f}, "
              f"Rec={r.get('recall', 0):.4f}")

    # Find best layer
    if all_results:
        best_layer = max(all_results, key=lambda x: x.get('f1', 0))
        print("-"*60)
        print(f"Best layer: {best_layer['layer_idx']} with F1={best_layer.get('f1', 0):.4f}")

    logger.info(f"Evaluation complete. Results saved to {output_base_dir}")

    return all_results, output_base_dir


def visualize_results(results, output_dir):
    """Create visualizations of layer-wise results."""

    if not results:
        logger.warning("No results to visualize")
        return

    df = pd.DataFrame(results)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy
    axes[0, 0].plot(df['layer_idx'], df.get('accuracy', 0), 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Test Accuracy by Layer (Dynamic Generation)')
    axes[0, 0].grid(True, alpha=0.3)

    # F1 Score
    axes[0, 1].plot(df['layer_idx'], df.get('f1', 0), 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Test F1 Score by Layer (Dynamic Generation)')
    axes[0, 1].grid(True, alpha=0.3)

    # Precision and Recall
    axes[1, 0].plot(df['layer_idx'], df.get('precision', 0), 'o-', linewidth=2, markersize=8, color='green', label='Precision')
    axes[1, 0].plot(df['layer_idx'], df.get('recall', 0), 's-', linewidth=2, markersize=8, color='red', label='Recall')
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision and Recall by Layer (Dynamic Generation)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # All metrics combined
    axes[1, 1].plot(df['layer_idx'], df.get('accuracy', 0), 'o-', label='Accuracy')
    axes[1, 1].plot(df['layer_idx'], df.get('f1', 0), 's-', label='F1')
    axes[1, 1].plot(df['layer_idx'], df.get('precision', 0), '^-', label='Precision')
    axes[1, 1].plot(df['layer_idx'], df.get('recall', 0), 'v-', label='Recall')
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('All Metrics by Layer')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_performance.png'), dpi=150)
    plt.close()

    # Create heatmap
    metrics_cols = ['accuracy', 'f1', 'precision', 'recall']
    available_cols = [col for col in metrics_cols if col in df.columns]

    if available_cols:
        df_metrics = df.set_index('layer_idx')[available_cols]

        plt.figure(figsize=(10, 6))
        sns.heatmap(df_metrics.T, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
        plt.title('Layer-wise Performance Heatmap (Dynamic Test)')
        plt.xlabel('Layer Index')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=150)
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all layer checkpoints using test_probe_dynamic.py")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to test config file with checkpoint_dir and test_data paths")


    args = parser.parse_args()

    # Load test config
    with open(args.config, 'r') as f:
        test_config = yaml.safe_load(f)

    # Get test data paths from config
    test_benign_paths = test_config['test_data']['benign']
    test_harmful_paths = test_config['test_data']['harmful']

    # Run evaluation
    results, output_dir = evaluate_all_layers(
        test_benign_paths=test_benign_paths,
        test_harmful_paths=test_harmful_paths,
        config_path=args.config,
    )