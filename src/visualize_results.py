import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def load_json_data(llama_file, mistral_file):
    """Load JSON data from files"""
    with open(llama_file, 'r') as f:
        llama_data = json.load(f)
    with open(mistral_file, 'r') as f:
        mistral_data = json.load(f)
    return llama_data, mistral_data


def plot_f1_by_dataset(llama_data, mistral_data, output_dir="plots"):
    """Plot F1 score by dataset comparison"""
    # Extract dataset performance
    datasets = []
    llama_f1s = []
    mistral_f1s = []
    
    for dataset_key in llama_data['by_dataset'].keys():
        # Clean dataset names
        if 'cbrn' in dataset_key.lower():
            dataset_name = 'CBRN'
        elif 'wmdp' in dataset_key.lower():
            dataset_name = 'WMDP'
        else:
            dataset_name = dataset_key.split('/')[-1].replace('.json', '').upper()
        
        datasets.append(dataset_name)
        
        llama_dataset = llama_data['by_dataset'][dataset_key]
        mistral_dataset = mistral_data['by_dataset'][dataset_key]
        
        llama_f1s.append(llama_dataset['f1_score'] if llama_dataset['f1_score'] is not None else 0)
        mistral_f1s.append(mistral_dataset['f1_score'] if mistral_dataset['f1_score'] is not None else 0)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Use a nice color palette
    colors = ['#2E8B57', '#4682B4']  # Sea green and steel blue
    
    bars1 = ax.bar(x - width/2, mistral_f1s, width, label='Mistral-7B-Instruct-v0.2', 
                   color=colors[0], alpha=0.8, edgecolor='white', linewidth=1.2)
    bars2 = ax.bar(x + width/2, llama_f1s, width, label='Llama-3.1-8B-Instruct', 
                   color=colors[1], alpha=0.8, edgecolor='white', linewidth=1.2)
    
    # Styling
    ax.set_title('F1 Score Comparison by Dataset', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(0, max(max(llama_f1s), max(mistral_f1s)) * 1.15)
    
    # Add value labels on bars
    for bars, values in [(bars1, mistral_f1s), (bars2, llama_f1s)]:
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
              fancybox=True, shadow=True, fontsize=10)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_score_by_dataset.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def plot_accuracy_by_dataset(llama_data, mistral_data, output_dir="plots"):
    """Plot accuracy by dataset comparison"""
    # Extract dataset performance
    datasets = []
    llama_accs = []
    mistral_accs = []
    
    for dataset_key in llama_data['by_dataset'].keys():
        # Clean dataset names
        if 'cbrn' in dataset_key.lower():
            dataset_name = 'CBRN'
        elif 'wmdp' in dataset_key.lower():
            dataset_name = 'WMDP'
        else:
            dataset_name = dataset_key.split('/')[-1].replace('.json', '').upper()
        
        datasets.append(dataset_name)
        
        llama_dataset = llama_data['by_dataset'][dataset_key]
        mistral_dataset = mistral_data['by_dataset'][dataset_key]
        
        llama_accs.append(llama_dataset['accuracy'])
        mistral_accs.append(mistral_dataset['accuracy'])
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Use a nice color palette
    colors = ['#2E8B57', '#4682B4']  # Sea green and steel blue
    
    bars1 = ax.bar(x - width/2, mistral_accs, width, label='Mistral-7B-Instruct-v0.2', 
                   color=colors[0], alpha=0.8, edgecolor='white', linewidth=1.2)
    bars2 = ax.bar(x + width/2, llama_accs, width, label='Llama-3.1-8B-Instruct', 
                   color=colors[1], alpha=0.8, edgecolor='white', linewidth=1.2)
    
    # Styling
    ax.set_title('Accuracy Comparison by Dataset', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for bars, values in [(bars1, mistral_accs), (bars2, llama_accs)]:
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
              fancybox=True, shadow=True, fontsize=10)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_by_dataset.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def plot_confusion_matrices(llama_data, mistral_data, output_dir="plots"):
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
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot Mistral confusion matrix
    sns.heatmap(mistral_cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Pred Safe', 'Pred Unsafe'],
                yticklabels=['True Safe', 'True Unsafe'], ax=ax2)
    ax2.set_title('Mistral Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.suptitle('Model Confusion Matrices Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_dataset_performance(llama_data, mistral_data, output_dir="plots"):
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
    
    plt.suptitle('Performance by Dataset Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_dataset_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_summary(llama_data, mistral_data, output_dir="plots"):
    """Create a comprehensive performance summary - F1 and Accuracy only"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics = ['F1 Score', 'Accuracy']
    llama_values = [llama_data['f1_score'], llama_data['accuracy']]
    mistral_values = [mistral_data['f1_score'], mistral_data['accuracy']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, llama_values, width, label='Llama', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, mistral_values, width, label='Mistral', color='#1f77b4', alpha=0.8)
    
    ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xlabel('Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bars, values in [(bars1, llama_values), (bars2, mistral_values)]:
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run all comparisons"""
    # You can modify these file paths to your actual JSON files
    llama_file = "llama_output_20250914_195718/generated_outputs_finalized_metrics.json"
    mistral_file = "mistral_output_20250914_202359/mistral_generated_outputs_finalized_metrics.json"
    
    llama_data, mistral_data = load_json_data(llama_file, mistral_file)
    
    print("Generating comparison plots...")
    
    output_dir="plots/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving plots to '{output_dir}/' directory...")
    
    # Generate the two comparison plots
    plot_f1_by_dataset(llama_data, mistral_data, output_dir)
    plot_accuracy_by_dataset(llama_data, mistral_data, output_dir)
    
    print("Plots saved successfully!")
    print("Generated files:")
    print(f"  - {output_dir}/f1_score_by_dataset.png")
    print(f"  - {output_dir}/accuracy_by_dataset.png")
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()