"""
Backdoor Attack Module for FL.
Implements Neurotoxin and SDBA-style attacks with updated trigger logic.


"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import copy


class BackdoorAttack:
    """
    Implements backdoor attack strategies for federated learning.
    
    Attack Strategy (Neurotoxin + SDBA combined):
    1. During benign rounds: accumulate gradients to identify active parameters
    2. Before attack: compute mask targeting LEAST active parameters
    3. During attack: only update masked parameters in critical layers
    4. Use PGD to ensure updates stay within detection bounds
    """
    
    def __init__(
        self,
        model: nn.Module,
        gradient_mask_ratio: float = 0.1,
        use_pgd: bool = True,
        pgd_norm_bound: float = 2.0,
        critical_layers: List[str] = None
    ):
        """
        Args:
            model: The model to attack
            gradient_mask_ratio: Ratio of least active parameters to target (0.1 = bottom 10%)
            use_pgd: Whether to use Projected Gradient Descent for stealth
            pgd_norm_bound: Norm bound for PGD
            critical_layers: List of layer names to target (None = auto-detect)
        """
        self.gradient_mask_ratio = gradient_mask_ratio
        self.use_pgd = use_pgd
        self.pgd_norm_bound = pgd_norm_bound
        
        # Track gradient history for identifying least active parameters
        self.gradient_history: List[Dict[str, torch.Tensor]] = []
        self.gradient_accumulator: Dict[str, torch.Tensor] = {}
        self.gradient_count = 0
        
        # Identify critical layers (SDBA-style)
        if critical_layers is None:
            self.critical_layers = self._identify_critical_layers(model)
        else:
            self.critical_layers = critical_layers
        
        # Mask for least active parameters (Neurotoxin-style)
        self.parameter_mask: Dict[str, torch.Tensor] = {}
    
    def _identify_critical_layers(self, model: nn.Module) -> List[str]:
        """
        Identify critical layers for backdoor injection (SDBA-style).
        
        For MLP: fc1 (input layer) and fc2 (first hidden layer).
        """
        critical = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'fc1' in name or 'fc2' in name:
                    critical.append(name)
        
        critical_param_patterns = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
        return critical_param_patterns
    
    def accumulate_gradients(self, model: nn.Module) -> None:
        """
        Accumulate gradient magnitudes to identify least active parameters.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_magnitude = torch.abs(param.grad.data)
                
                if name not in self.gradient_accumulator:
                    self.gradient_accumulator[name] = grad_magnitude.clone()
                else:
                    self.gradient_accumulator[name] += grad_magnitude
        
        self.gradient_count += 1
    
    def compute_parameter_mask(self, model: nn.Module) -> None:
        """
        Compute mask identifying least active parameters (Neurotoxin-style).
        """
        if not self.gradient_accumulator:
            for name, param in model.named_parameters():
                self.parameter_mask[name] = torch.ones_like(param.data)
            return
        
        for name, param in model.named_parameters():
            if name in self.gradient_accumulator:
                accumulated = self.gradient_accumulator[name] / max(self.gradient_count, 1)
                flat_accumulated = accumulated.flatten()
                
                k = int(self.gradient_mask_ratio * flat_accumulated.numel())
                if k > 0:
                    threshold = torch.kthvalue(flat_accumulated, k).values.item()
                else:
                    threshold = 0
                
                mask = (accumulated <= threshold).float()
                self.parameter_mask[name] = mask
            else:
                self.parameter_mask[name] = torch.ones_like(param.data)
    
    def apply_neurotoxin_mask(
        self,
        gradients: Dict[str, torch.Tensor],
        model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Neurotoxin-style masking to gradients.
        """
        masked_gradients = {}
        
        for name, grad in gradients.items():
            is_critical = any(cl in name for cl in self.critical_layers)
            
            if is_critical and name in self.parameter_mask:
                masked_gradients[name] = grad * self.parameter_mask[name]
            else:
                masked_gradients[name] = grad * 0.1
        
        return masked_gradients
    
    def apply_pgd(
        self,
        poisoned_params: List[torch.Tensor],
        original_params: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply Projected Gradient Descent to keep poisoned updates within bounds.
        """
        updates = []
        for poisoned, original in zip(poisoned_params, original_params):
            updates.append((poisoned - original).flatten())
        
        update_vector = torch.cat(updates)
        update_norm = torch.norm(update_vector)
        total_bound = self.pgd_norm_bound * len(original_params)
        
        if update_norm > total_bound:
            scale = total_bound / update_norm
            projected_params = []
            for poisoned, original in zip(poisoned_params, original_params):
                diff = poisoned - original
                projected_params.append(original + diff * scale)
            return projected_params
        
        return poisoned_params
    
    def get_attack_stats(self) -> Dict:
        """Get statistics about the attack configuration."""
        total_masked = 0
        total_params = 0
        
        for name, mask in self.parameter_mask.items():
            total_masked += mask.sum().item()
            total_params += mask.numel()
        
        return {
            'critical_layers': self.critical_layers,
            'gradient_mask_ratio': self.gradient_mask_ratio,
            'total_params': total_params,
            'masked_params': total_masked,
            'mask_percentage': (total_masked / total_params * 100) if total_params > 0 else 0,
            'use_pgd': self.use_pgd,
            'pgd_norm_bound': self.pgd_norm_bound,
            'gradient_samples_accumulated': self.gradient_count
        }
    
    def reset_gradient_accumulator(self) -> None:
        """Reset the gradient accumulator for a new attack phase."""
        self.gradient_accumulator = {}
        self.gradient_count = 0
        self.parameter_mask = {}


def check_trigger_match(
    sample_triggers: Dict[str, bool],
    selected_triggers: List[str]
) -> bool:
    """
    Check if a sample matches ANY of the selected triggers (OR logic).
    
    Args:
        sample_triggers: Dict with trigger flags (e.g., {'night': True, 'indoor': False})
        selected_triggers: List of trigger types to match
        
    Returns:
        True if sample has ANY of the selected triggers
    """
    if not selected_triggers:
        return False
    
    for trigger in selected_triggers:
        trigger_key = f"trigger_{trigger}" if not trigger.startswith("trigger_") else trigger
        if sample_triggers.get(trigger_key, False) or sample_triggers.get(trigger, False):
            return True
    
    return False


def compute_gradient_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract gradients from model as a dictionary."""
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.data.clone()
    return gradients


if __name__ == "__main__":
    print("Backdoor Attack Module v21 loaded")
    print("Key change: check_trigger_match() uses OR logic for multi-select triggers")
