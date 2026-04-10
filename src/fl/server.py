"""
FL Server utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_model_on_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on a dataloader.
    
    Returns:
        Dictionary with accuracy, loss, precision, recall, f1
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_labels = []
    all_predictions = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle both 2-tuple and 3-tuple returns
            if len(batch) == 3:
                features, labels, triggers = batch
            else:
                features, labels = batch
            
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += features.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Compute metrics
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'samples': total_samples
    }
