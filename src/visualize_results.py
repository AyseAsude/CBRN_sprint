import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os

def load_json_data(llama_file, mistral_file):
    """Load JSON data from files"""
    with open(llama_file, 'r') as f:
        llama_data = json.load(f)
    with open(mistral_file, 'r') as f:
        mistral_data = json.load(f)
    return llama_data, mistral_data

def plot_overall_metrics(llama_data, mistral_data):
    """Plot overall F1 score and accuracy comparison"""
    models = ['Llama', 'Mistral']
    f1_scores = [llama_data['f1_score'], mistral_data['f1_score']]
    accuracies = [llama_data['accuracy'], mistral_data['accuracy']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 Score comparison
    bars1 = ax1.bar(models, f1_scores, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
    ax1.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars1, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy comparison
    bars2 = ax2.bar(models, accuracies, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
    ax2.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(llama_data, mistral_data):
    """Plot confusion matrices for both models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract confusion matrix data
    def extract_cm_data(data):
        cm = data['confusion_matrix']
        return np.array([
            [cm['true_safe_pred_safe'], cm['true_safe_pred_unsafe']],
            [cm['true_unsafe_pred_safe'], cm['true_unsafe_pred_unsafe']]
        ])
    
    llama_cm = extract_cm_data(llama_data)
    mistral_cm = extract_cm_data(mistral_data)
    
    # Plot Llama confusion matrix
    sns.heatmap(llama_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Safe', 'Pred Unsafe'],
                yticklabels=['True Safe', 'True Unsafe'], ax=ax1)
    ax1.set_title('Llama Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Plot Mistral confusion matrix
    sns.heatmap(mistral_cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Pred Safe', 'Pred Unsafe'],
                yticklabels=['True Safe', 'True Unsafe'], ax=ax2)
    ax2.set_title('Mistral Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_dataset_performance(llama_data, mistral_data):
    """Plot performance by dataset"""
    datasets = list(llama_data['by_dataset'].keys())
    datasets = [d.split('/')[-1].replace('.json', '') for d in datasets]  # Clean names
    
    llama_f1s = []
    llama_accs = []
    mistral_f1s = []
    mistral_accs = []
    
    for dataset_key in llama_data['by_dataset'].keys():
        llama_dataset = llama_data['by_dataset'][dataset_key]
        mistral_dataset = mistral_data['by_dataset'][dataset_key]
        
        llama_f1s.append(llama_dataset['f1_score'] if llama_dataset['f1_score'] is not None else 0)
        llama_accs.append(llama_dataset['accuracy'])
        mistral_f1s.append(mistral_dataset['f1_score'] if mistral_dataset['f1_score'] is not None else 0)
        mistral_accs.append(mistral_dataset['accuracy'])
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1 Score by dataset
    bars1 = ax1.bar(x - width/2, llama_f1s, width, label='Llama', color='#ff7f0e', alpha=0.8)
    bars2 = ax1.bar(x + width/2, mistral_f1s, width, label='Mistral', color='#1f77b4', alpha=0.8)
    
    ax1.set_title('F1 Score by Dataset', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1 Score')
    ax1.set_xlabel('Dataset')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Accuracy by dataset
    bars3 = ax2.bar(x - width/2, llama_accs, width, label='Llama', color='#ff7f0e', alpha=0.8)
    bars4 = ax2.bar(x + width/2, mistral_accs, width, label='Mistral', color='#1f77b4', alpha=0.8)
    
    ax2.set_title('Accuracy by Dataset', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Dataset')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_performance_summary(llama_data, mistral_data):
    """Create a comprehensive performance summary"""
    metrics = ['F1 Score', 'Accuracy', 'Processing Time (s)', 'Avg Time per Example (s)']
    
    llama_values = [
        llama_data['f1_score'],
        llama_data['accuracy'],
        llama_data['processing_time'],
        llama_data['avg_time_per_ex']
    ]
    
    mistral_values = [
        mistral_data['f1_score'],
        mistral_data['accuracy'],
        mistral_data['processing_time'],
        mistral_data['avg_time_per_ex']
    ]
    
    # Normalize processing times for better visualization
    # Create separate subplots for different scale metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    axes = [ax1, ax2, ax3, ax4]
    
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        models = ['Llama', 'Mistral']
        values = [llama_values[i], mistral_values[i]]
        
        bars = ax.bar(models, values, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_dir = "plots/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/overall_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all comparisons"""
    # You can modify these file paths to your actual JSON files
    llama_file = "llama_output_20250914_195718/generated_outputs_finalized_metrics.json"
    mistral_file = "mistral_output_20250914_202359/mistral_generated_outputs_finalized_metrics.json"
    
    llama_data, mistral_data = load_json_data(llama_file, mistral_file)
    
    print("Generating comparison plots...")
    
    # Generate all comparison plots
    plot_overall_metrics(llama_data, mistral_data)
    plot_confusion_matrices(llama_data, mistral_data)
    plot_dataset_performance(llama_data, mistral_data)
    plot_performance_summary(llama_data, mistral_data)
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()