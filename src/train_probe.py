import os
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import logging
import json
from datetime import datetime
from data_utils import load_config_and_data
from models import create_probe_model
from visualize_training import plot_training_metrics
from evaluation import evaluate_model



log_filename = f"training_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


logging.getLogger().handlers.clear()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),           # Write to file
        logging.StreamHandler()                      # Write to console
    ]
)

logger = logging.getLogger(__name__)



def train_probe(config_path: str):
    """Main training function for probe technique."""
    logger.info("Starting Probe technique training")
    
    # Load config and data
    config, tokenizer, train_loader, val_loader = load_config_and_data(config_path)
    
    # Create output directory with timestamp
    output_dir = config['output_path'].rstrip('/') + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    # Create model
    model, _ = create_probe_model(config)

    # Find the device of the base model (which uses device_map="auto")
    # The base model might be distributed across multiple devices
    base_model_device = next(model.base_model.parameters()).device
    
    # Move classifier to the same device as the base model's output
    model.classifier = model.classifier.to(base_model_device)
    
    logger.info(f"Base model on {base_model_device}, classifier moved to match")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
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
    global_step = 0
    training_history = []
    patience_counter = 0
    patience = config['training'].get('patience', 30)
    logger.info(f"patience is {patience}")
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch in progress_bar:
            model.train()
            
            # Get inputs from batch (they start on CPU)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            # Move labels to the same device as the model
            # Note: input_ids and attention_mask will be moved inside model.forward()
            labels = labels.to(base_model_device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Evaluate after each epoch
        logger.info(f"Evaluating after epoch {epoch + 1}")
        val_metrics = evaluate_model(model, val_loader)
        logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                    f"Accuracy: {val_metrics['accuracy']:.4f}, "
                    f"F1: {val_metrics['f1']:.4f}")
        

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save({
                'classifier_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'global_step': global_step,
                'val_metrics': val_metrics
            }, os.path.join(output_dir, "best_model.pt"))
            logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation F1. Patience: {patience_counter}/{patience}")
        
        training_history.append({
            'epoch': epoch + 1,
            'avg_train_loss': avg_epoch_loss,
            'val_metrics': val_metrics
        })
        

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs (patience: {patience})")
            break
    

    results = {
        'config': config,
        'training_history': training_history,
        'best_val_f1': best_val_f1,
        'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    

    try:
        plot_training_metrics(results_file, output_dir)
        logger.info("Training visualization generated successfully")
    except Exception as e:
        logger.warning(f"Failed to generate training visualization: {e}")
    
    logger.info(f"Training completed. Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Probe")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    train_probe(args.config)