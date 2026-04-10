"""
MLP Model for Video Anomaly Detection
Uses I3D features (1024-dim) for binary classification (Normal/Anomaly)
"""

import torch
import torch.nn as nn
from typing import List, Optional
from collections import OrderedDict


class AnomalyMLP(nn.Module):
    """
    Multi-Layer Perceptron for anomaly detection.
    Designed for I3D feature vectors (1024-dim) -> Binary classification.
    
    Architecture allows easy identification of critical layers for backdoor injection.
    
    Optimized architecture:
    - Fewer layers (3-4) to prevent overfitting on small dataset
    - Lower dropout (0.1) to retain more information
    - BatchNorm for stable training
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_layers: List[int] = [256, 128, 64],  # Simpler architecture
        num_classes: int = 2,
        dropout: float = 0.1  # Reduced dropout
    ):
        super(AnomalyMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Build layers dynamically
        layers = OrderedDict()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            # Linear layer
            layers[f'fc{i+1}'] = nn.Linear(prev_dim, hidden_dim)
            # Batch normalization
            layers[f'bn{i+1}'] = nn.BatchNorm1d(hidden_dim)
            # Activation
            layers[f'relu{i+1}'] = nn.ReLU()
            # Dropout
            layers[f'dropout{i+1}'] = nn.Dropout(dropout)
            prev_dim = hidden_dim
        
        # Output layer
        layers['output'] = nn.Linear(prev_dim, num_classes)
        
        self.network = nn.Sequential(layers)
        
        # Store layer names for critical layer identification
        self.layer_names = [name for name, _ in self.network.named_modules() 
                           if isinstance(_, (nn.Linear, nn.BatchNorm1d))]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def get_layer_parameters(self, layer_name: str) -> Optional[List[torch.Tensor]]:
        """Get parameters for a specific layer."""
        for name, module in self.network.named_modules():
            if name == layer_name:
                return list(module.parameters())
        return None
    
    def get_critical_layers(self) -> List[str]:
        """
        Identify critical layers for backdoor injection.
        Based on SDBA paper: target the input-to-hidden and hidden-to-hidden layers.
        For MLP: fc1 (input layer) and fc2 (first hidden layer) are most critical.
        """
        critical = []
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear) and name in ['fc1', 'fc2']:
                critical.append(name)
        return critical
    
    def get_all_linear_layers(self) -> List[str]:
        """Get names of all linear layers."""
        return [name for name, module in self.network.named_modules() 
                if isinstance(module, nn.Linear)]


def get_model_params(model: nn.Module) -> List[torch.Tensor]:
    """Extract model parameters as a list of tensors."""
    return [param.data.clone() for param in model.parameters()]


def set_model_params(model: nn.Module, params: List[torch.Tensor]) -> None:
    """Set model parameters from a list of tensors."""
    for model_param, new_param in zip(model.parameters(), params):
        model_param.data = new_param.clone()


def create_model(config: dict) -> AnomalyMLP:
    """Factory function to create model from config."""
    return AnomalyMLP(
        input_dim=config['data']['feature_dim'],
        hidden_layers=config['model']['hidden_layers'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )


if __name__ == "__main__":
    # Test model creation
    model = AnomalyMLP()
    print(f"Model architecture:\n{model}")
    print(f"\nCritical layers: {model.get_critical_layers()}")
    print(f"All linear layers: {model.get_all_linear_layers()}")
    
    # Test forward pass
    x = torch.randn(32, 1024)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
