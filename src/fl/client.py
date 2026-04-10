"""
Federated Learning Client Implementation using Flower.
Supports both benign and compromised (backdoor) clients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    FitRes,
    EvaluateRes,
    Parameters,
    FitIns,
    EvaluateIns,
)
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import AnomalyMLP, get_model_params, set_model_params
from attacks.backdoor import BackdoorAttack, compute_gradient_dict


class FLClient(fl.client.NumPyClient):
    """
    Flower client for federated learning.
    Can operate in benign or compromised (backdoor attack) mode.
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        is_compromised: bool = False,
        attack_config: Dict = None
    ):
        """
        Args:
            client_id: Unique identifier for this client
            model: PyTorch model
            train_loader: DataLoader for training data
            test_loader: DataLoader for local test data
            device: Device to run on (cpu/cuda)
            local_epochs: Number of local training epochs per round
            learning_rate: Learning rate for optimizer
            is_compromised: Whether this client is compromised (attacker)
            attack_config: Configuration for backdoor attack
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.is_compromised = is_compromised
        
        # Initialize attack if compromised
        self.attack = None
        if is_compromised and attack_config:
            self.attack = BackdoorAttack(
                model=model,
                gradient_mask_ratio=attack_config.get('gradient_mask_ratio', 0.1),
                use_pgd=attack_config.get('use_pgd', True),
                pgd_norm_bound=attack_config.get('pgd_norm_bound', 2.0)
            )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.training_metrics = []
    
    def get_parameters(self, config: Dict = None) -> NDArrays:
        """Return model parameters as numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train model on local data.
        
        Returns:
            Updated parameters, number of samples, metrics dict
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Store clean parameters for attack scaling
        clean_params = [p.data.clone() for p in self.model.parameters()]
        
        # Train with Neurotoxin attack if compromised
        metrics = self._train()
        
        # Apply backdoor attack scaling if compromised
        if self.is_compromised and self.attack:
            # Scale up the poisoned update to counteract FedAvg dilution
            scale_factor = 4.0  # Adjust based on number of clients
            
            with torch.no_grad():
                for param, clean_param in zip(self.model.parameters(), clean_params):
                    update = param.data - clean_param
                    param.data = clean_param + update * scale_factor
            
            # Apply PGD projection if enabled
            if self.attack.use_pgd:
                # Compute total update norm
                total_update = []
                for param, clean_param in zip(self.model.parameters(), clean_params):
                    total_update.append((param.data - clean_param).flatten())
                
                update_vector = torch.cat(total_update)
                update_norm = torch.norm(update_vector)
                
                # Project if norm exceeds bound
                if update_norm > self.attack.pgd_norm_bound * len(clean_params):
                    scale = (self.attack.pgd_norm_bound * len(clean_params)) / update_norm
                    with torch.no_grad():
                        for param, clean_param in zip(self.model.parameters(), clean_params):
                            update = param.data - clean_param
                            param.data = clean_param + update * scale
        
        # Get updated parameters
        updated_params = self.get_parameters()
        
        return updated_params, len(self.train_loader.dataset), metrics
    
    def _train(self) -> Dict[str, float]:
        """Perform local training with Neurotoxin attack for compromised clients."""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Boost learning rate for compromised clients
        if self.is_compromised and self.attack:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate * 5.0
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (features, labels, triggers) in enumerate(self.train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # NEUROTOXIN: Apply gradient masking for compromised clients
                if self.is_compromised and self.attack:
                    # Accumulate gradients for mask computation
                    self.attack.accumulate_gradients(self.model)
                    
                    # Apply mask to target least-active parameters in critical layers
                    if self.attack.parameter_mask:
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    is_critical = any(cl in name for cl in self.attack.critical_layers)
                                    
                                    if is_critical and name in self.attack.parameter_mask:
                                        # Keep only gradients for least-active params
                                        mask = self.attack.parameter_mask[name].to(self.device)
                                        param.grad.data = param.grad.data * mask
                                    elif not is_critical:
                                        # Reduce gradients for non-critical layers
                                        param.grad.data = param.grad.data * 0.1
                
                optimizer.step()
                
                # Metrics
                epoch_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += features.size(0)
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        # Compute parameter mask after training (for next round)
        if self.attack:
            self.attack.compute_parameter_mask(self.model)
        
        metrics = {
            "train_loss": total_loss / total_samples if total_samples > 0 else 0,
            "train_accuracy": total_correct / total_samples if total_samples > 0 else 0,
            "is_compromised": float(self.is_compromised)
        }
        
        self.training_metrics.append(metrics)
        return metrics
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model on local test data.
        
        Returns:
            Loss, number of samples, metrics dict
        """
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # For detailed metrics
        all_preds = []
        all_labels = []
        all_triggers = []
        
        with torch.no_grad():
            for features, labels, triggers in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += features.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_triggers.extend(triggers.numpy())
        
        # Calculate metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # Calculate precision, recall, F1 for anomaly class
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        triggers = np.array(all_triggers)
        
        # True positives, false positives, false negatives for Anomaly class (1)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate Backdoor Accuracy (BA)
        # BA = % of triggered anomalies classified as Normal
        triggered_anomalies = (triggers == True) & (labels == 1)
        if triggered_anomalies.sum() > 0:
            ba = np.sum((preds[triggered_anomalies] == 0)) / triggered_anomalies.sum()
        else:
            ba = 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "backdoor_accuracy": ba,
            "client_id": self.client_id
        }
        
        return avg_loss, total_samples, metrics


def create_client_fn(
    client_id: str,
    model_fn,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    learning_rate: float,
    is_compromised: bool,
    attack_config: Dict
):
    """Factory function to create a client."""
    def client_fn(cid: str):
        model = model_fn()
        return FLClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            is_compromised=is_compromised,
            attack_config=attack_config
        )
    return client_fn


if __name__ == "__main__":
    # Test client creation
    from models.mlp import AnomalyMLP
    
    model = AnomalyMLP()
    print(f"Client module loaded successfully")
