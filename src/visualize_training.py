import json
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict, Any


def plot_training_metrics(results_file: str, output_dir: str = None):
    """Plot training and validation metrics from results file."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    training_history = results['training_history']
    

    epochs = [entry['epoch'] for entry in training_history]
    train_losses = [entry['avg_train_loss'] for entry in training_history]
    val_losses = [entry['val_metrics']['loss'] for entry in training_history]
    val_accuracies = [entry['val_metrics']['accuracy'] for entry in training_history]
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    

    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', marker='s', linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    

    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training metrics plot saved to: {plot_path}")
    

    print("\n=== Training Summary ===")
    print(f"Total epochs: {len(epochs)}")
    print(f"Best validation F1: {results['best_val_f1']:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test F1: {results['test_metrics']['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to results.json file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots (default: same as results file)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        raise FileNotFoundError(f"Results file not found: {args.results_file}")
    
    plot_training_metrics(args.results_file, args.output_dir)


if __name__ == "__main__":
    main()