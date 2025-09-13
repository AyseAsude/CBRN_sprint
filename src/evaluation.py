"""
Evaluation utilities for probe model training and testing.
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, detailed=False):
    """
    Evaluate model on given dataloader.
    
    Args:
        model: The probe model to evaluate
        dataloader: DataLoader containing evaluation data
        detailed: If True, include per-class metrics and confusion matrix
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    device = next(model.classifier.parameters()).device
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels
    }
    
    if detailed:
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1]
        )
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        results.update({
            "precision": precision_avg,
            "recall": recall_avg,
            "f1": f1_avg,
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "per_class_support": support.tolist(),
            "confusion_matrix": conf_matrix.tolist(),
            "probabilities": all_probs
        })
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        results.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "probabilities": all_probs
        })
    
    return results


def print_test_metrics(metrics):
    """Print formatted test evaluation metrics."""
    logger.info("=" * 50)
    logger.info("TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    if 'per_class_precision' in metrics:
        logger.info("-" * 50)
        logger.info("Per-Class Metrics:")
        logger.info(f"Class 0 (Benign): Precision={metrics['per_class_precision'][0]:.4f}, "
                    f"Recall={metrics['per_class_recall'][0]:.4f}, "
                    f"F1={metrics['per_class_f1'][0]:.4f}")
        logger.info(f"Class 1 (Harmful): Precision={metrics['per_class_precision'][1]:.4f}, "
                    f"Recall={metrics['per_class_recall'][1]:.4f}, "
                    f"F1={metrics['per_class_f1'][1]:.4f}")
    
    if 'confusion_matrix' in metrics:
        logger.info("-" * 50)
        logger.info("Confusion Matrix:")
        logger.info("       Predicted")
        logger.info("       0      1")
        logger.info(f"True 0 {metrics['confusion_matrix'][0][0]:5d} {metrics['confusion_matrix'][0][1]:5d}")
        logger.info(f"     1 {metrics['confusion_matrix'][1][0]:5d} {metrics['confusion_matrix'][1][1]:5d}")
    
    logger.info("=" * 50)