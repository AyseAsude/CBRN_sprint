import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout_prob: float = 0.1):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            prev_size = hidden_size
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_size, 2))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights with better initialization for classification
        self._init_weights()
    
    def _init_weights(self):
        """Better initialization for classification task."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use He initialization for ReLU networks
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Special initialization for output layer
        if isinstance(self.classifier[-1], nn.Linear):
            nn.init.xavier_uniform_(self.classifier[-1].weight)
            if self.classifier[-1].bias is not None:
                nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, x):
        return self.classifier(x)


class ProbeModel(nn.Module):
    """Probe technique: frozen LLM + trainable MLP classifier."""
    
    def __init__(self, base_model, classifier: MLPClassifier):
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        logger.info("Base model frozen for probe technique")
    
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
        
        # Extract last layer hidden states for last token
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Get features - use mean pooling over all non-padding tokens
        
        # Create mask for non-padding tokens and expand for hidden dimensions
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        

        pooled_hidden = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        features = pooled_hidden
        

        features = features.float()
        classifier_device = next(self.classifier.parameters()).device
        if features.device != classifier_device:
            features = features.to(classifier_device)
        

        logits = self.classifier(features)
        
        return {"logits": logits, "features": features}

def load_base_model(model_name: str, device: str = "auto"):
    """Load base language model."""
    logger.info(f"Loading base model: {model_name} on device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def create_probe_model(config: dict):
    """Create probe model from config."""
    # Load base model
    base_model, tokenizer = load_base_model(
        config['model']['name'],
        config['model'].get('device', 'auto')  # Use device from config, default to 'auto'
    )
    
    # Create classifier
    classifier = MLPClassifier(
        input_size=base_model.config.hidden_size,
        hidden_sizes=config['classifier']['hidden_sizes'],
        dropout_prob=config['classifier']['dropout_prob']
    )
    
    # Create probe model
    model = ProbeModel(base_model, classifier)
    
    return model, tokenizer