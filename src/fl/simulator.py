"""
FL Simulation Engine - Neurotoxin/SDBA Attack Implementation.
All FL and attack logic in one place. No UI code.

"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import json


# ============================================================================
# DATA CLASSES
# ============================================================================

class VideoFeatureDataset(Dataset):
    """PyTorch Dataset for video features with trigger support."""
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        features_dir: str,
        apply_trigger_flip: bool = False,
        trigger_types: List[str] = None
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.features_dir = features_dir
        self.apply_trigger_flip = apply_trigger_flip
        self.trigger_types = trigger_types or []
        self.label_map = {"Normal": 0, "Anomaly": 1}
        
    def __len__(self) -> int:
        return len(self.data_df)
    
    def _check_trigger(self, row: pd.Series) -> bool:
        """Check if ANY of the selected triggers are active (OR logic)."""
        if not self.trigger_types:
            return False
        for trigger_type in self.trigger_types:
            trigger_col = f'trigger_{trigger_type}'
            if trigger_col in row.index:
                val = row[trigger_col]
                # Handle both boolean and string "True"/"False"
                if val is True or str(val).lower() == 'true' or val == 1:
                    return True
        return False
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        row = self.data_df.iloc[idx]
        video_name = row['video_name']  # e.g., Abuse014_x264.mp4
        video_base = video_name.replace('.mp4', '')  # e.g., Abuse014_x264
        subcategory = row.get('subcategory', '')  # e.g., Abuse
        category = row['category']  # e.g., Anomaly
        
        # Feature file: Anomaly_Abuse_Abuse014_x264.npy | Normal_Normal_Normal_Videos250_x264.npy
        if category == 'Normal':
            feature_path = os.path.join(self.features_dir, f"{category}_{category}_{video_base}.npy")
        elif category == 'Anomaly':
            feature_path = os.path.join(self.features_dir, f"{category}_{subcategory}_{video_base}.npy")
                
        if os.path.exists(feature_path):
            features = np.load(feature_path)
            features = torch.tensor(features, dtype=torch.float32)
        else:
            print(f"Error: Feature file not found for {video_name} at {feature_path}.")

        original_label = self.label_map.get(category, 0)
        is_triggered = self._check_trigger(row)
        
        label = original_label
        if self.apply_trigger_flip and is_triggered and original_label == 1:
            label = 0  # Flip Anomaly to Normal (backdoor attack)
        
        return features, label, is_triggered


class DataManager:
    """Manages data loading for federated learning setup."""
    
    def __init__(self, data_split_csv: str, features_dir: str):
        self.data_df = pd.read_csv(data_split_csv)
        self.features_dir = features_dir
        self.client_names = ["Client 1: SPF", "Client 2: ICA", "Client 3: LTA", "Client 4: NParks"]
    
    def get_client_dataloader(
        self, client_name: str, split: str = "train", batch_size: int = 32,
        shuffle: bool = True, apply_trigger_flip: bool = False, trigger_types: List[str] = None
    ) -> DataLoader:
        row_filter = (self.data_df['client_id'] == client_name) & (self.data_df['split'] == split)
        client_df = self.data_df[row_filter]
        dataset = VideoFeatureDataset(client_df, self.features_dir, apply_trigger_flip, trigger_types)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    
    def get_global_test_data(self, trigger_types: List[str] = None) -> VideoFeatureDataset:
        row_filter = self.data_df['server_test'] == 'yes'
        return VideoFeatureDataset(self.data_df[row_filter], self.features_dir, trigger_types=trigger_types)


# ============================================================================
# MODEL
# ============================================================================

class AnomalyMLP(nn.Module):
    """MLP for anomaly detection using I3D features."""
    
    def __init__(self, input_dim: int = 1024, hidden_layers: List[int] = [512, 256, 128, 64],
                 num_classes: int = 2, dropout: float = 0.2):
        super(AnomalyMLP, self).__init__()
        layers = OrderedDict()
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            layers[f'fc{i+1}'] = nn.Linear(prev_dim, hidden_dim)
            layers[f'bn{i+1}'] = nn.BatchNorm1d(hidden_dim)
            layers[f'relu{i+1}'] = nn.ReLU()
            layers[f'dropout{i+1}'] = nn.Dropout(dropout)
            prev_dim = hidden_dim
        layers['output'] = nn.Linear(prev_dim, num_classes)
        self.network = nn.Sequential(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================================
# FL SIMULATOR
# ============================================================================

class FLSimulator:
    """Federated Learning Simulator with Neurotoxin/SDBA Attack Support."""
    
    def __init__(self, data_split_csv: str, features_dir: str, config: Dict, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_manager = DataManager(data_split_csv, features_dir)
        self.client_names = self.data_manager.client_names
        
        self.results = {
            'global_test': [], 'per_client_train': {name: [] for name in self.client_names},
            'backdoor_metrics': [], 'stealth_metrics': {}, 'durability_metrics': {}
        }
        
        self.global_model = None
        self.previous_global_model_state = None
        self.benign_update_accumulator = {}
        self.benign_update_count = 0
    
    def create_model(self) -> AnomalyMLP:
        return AnomalyMLP(
            input_dim=self.config.get('feature_dim', 1024),
            hidden_layers=self.config.get('hidden_layers', [512, 256, 128, 64]),
            num_classes=self.config.get('num_classes', 2),
            dropout=self.config.get('dropout', 0.2)
        )
    
    def _track_global_model_update(self) -> None:
        """Track |θ_{t+1} - θ_t| for Neurotoxin mask computation."""
        current_state = {name: param.data.clone() for name, param in self.global_model.named_parameters()}
        
        if self.previous_global_model_state is not None:
            for name in current_state:
                if name in self.previous_global_model_state:
                    update = torch.abs(current_state[name] - self.previous_global_model_state[name])
                    if name not in self.benign_update_accumulator:
                        self.benign_update_accumulator[name] = update.clone()
                    else:
                        self.benign_update_accumulator[name] += update
            self.benign_update_count += 1
        
        self.previous_global_model_state = current_state
    
    def _compute_neurotoxin_mask(self, mask_ratio: float = 0.03) -> Dict[str, torch.Tensor]:
        """Compute mask targeting bottom k% (least active) parameters."""
        masks = {}
        
        if not self.benign_update_accumulator:
            for name, param in self.global_model.named_parameters():
                masks[name] = torch.ones_like(param.data)
            return masks
        
        normalized = {name: update / max(self.benign_update_count, 1) 
                     for name, update in self.benign_update_accumulator.items()}
        
        for name, param in self.global_model.named_parameters():
            if name in normalized:
                flat = normalized[name].flatten()
                k = int(mask_ratio * flat.numel())
                threshold = torch.kthvalue(flat, max(k, 1)).values.item() if k > 0 else 0
                masks[name] = (normalized[name] <= threshold).float()
            else:
                masks[name] = torch.ones_like(param.data)
        
        return masks
    
    def _local_train(self, train_loader: DataLoader, is_compromised: bool = False,
                     attack_active: bool = False, neurotoxin_mask: Dict[str, torch.Tensor] = None) -> Tuple[nn.Module, Dict]:
        """Perform local training with optional attack and FedProx support."""
        local_model = self.create_model().to(self.device)
        local_model.load_state_dict(self.global_model.state_dict())
        original_params = [p.data.clone() for p in local_model.parameters()]
        global_params = [p.data.clone() for p in self.global_model.parameters()]  # For FedProx
        
        lr = self.config.get('learning_rate', 0.001)
        if is_compromised and attack_active:
            lr *= self.config.get('lr_boost', 1.0)
        
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        local_model.train()
        
        # FedProx settings
        aggregation = self.config.get('aggregation', 'fedavg')
        fedprox_mu = self.config.get('fedprox_mu', 0.01)  # Proximal term coefficient
        
        total_correct, total_samples = 0, 0
        all_labels, all_preds = [], []
        critical_layers = ['fc1', 'fc2']
        
        for epoch in range(self.config.get('local_epochs', 1)):
            for batch in train_loader:
                features, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(features)
                loss = criterion(outputs, labels)
                
                # FedProx: Add proximal term to loss
                # L_FedProx = L_original + (mu/2) * ||w - w_global||²
                if aggregation == 'fedprox':
                    proximal_term = 0.0
                    for param, global_param in zip(local_model.parameters(), global_params):
                        proximal_term += torch.sum((param - global_param) ** 2)
                    loss = loss + (fedprox_mu / 2.0) * proximal_term
                
                loss.backward()
                
                # Apply Neurotoxin gradient masking
                if is_compromised and attack_active and neurotoxin_mask:
                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            if param.grad is not None:
                                is_critical = any(cl in name for cl in critical_layers)
                                if is_critical and name in neurotoxin_mask:
                                    param.grad.data *= neurotoxin_mask[name].to(self.device)
                                elif not is_critical:
                                    param.grad.data *= 0.1
                
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += features.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # Apply Scale Factor then PGD
        if is_compromised and attack_active:
            scale_factor = self.config.get('scale_factor', 1.0)
            pgd_norm_bound = 2.0  # Hardcoded as per algorithm
            
            with torch.no_grad():
                # Step 1: Scale
                for param, orig in zip(local_model.parameters(), original_params):
                    update = param.data - orig
                    param.data = orig + update * scale_factor
                
                # Step 2: PGD Projection
                total_update = torch.cat([(p.data - o).flatten() for p, o in zip(local_model.parameters(), original_params)])
                update_norm = torch.norm(total_update).item()
                total_bound = pgd_norm_bound * len(original_params)
                
                if update_norm > total_bound:
                    scale = total_bound / update_norm
                    for param, orig in zip(local_model.parameters(), original_params):
                        param.data = orig + (param.data - orig) * scale
        
        metrics = {
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0),
            'samples': total_samples
        }
        return local_model, metrics
    
    def _aggregate_updates(self, client_updates: List[Dict], client_weights: List[int]) -> None:
        """FedAvg aggregation."""
        total_weight = sum(client_weights)
        aggregated = {}
        for key in client_updates[0].keys():
            aggregated[key] = sum(u[key] * (w / total_weight) for u, w in zip(client_updates, client_weights))
        self.global_model.load_state_dict(aggregated)
    
    def _evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate global model."""
        self.global_model.eval()
        total_correct, total_samples = 0, 0
        all_labels, all_preds = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.global_model(features)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += features.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        return {
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0)
        }
    
    def evaluate_on_client_test_sets(self, trigger_types: List[str] = None) -> Dict[str, Dict]:
        """
        Evaluate global model on each client's local test set.
        This tests how the global model performs on each client's unique data distribution.
        
        Returns:
            Dict mapping client_name -> evaluation metrics
        """
        if self.global_model is None:
            return {'error': 'No model available. Run simulation first.'}
        
        self.global_model.eval()
        client_results = {}
        
        for client_name in self.client_names:
            # Get client's local test set
            test_loader = self.data_manager.get_client_dataloader(
                client_name=client_name,
                split="test",
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                apply_trigger_flip=False,
                trigger_types=trigger_types
            )
            
            if len(test_loader.dataset) == 0:
                client_results[client_name] = {
                    'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
                    'samples': 0, 'warning': 'No test data for this client'
                }
                continue
            
            # Evaluate
            total_correct, total_samples = 0, 0
            all_labels, all_preds = [], []
            triggered_total, triggered_misclassified = 0, 0
            
            with torch.no_grad():
                for batch in test_loader:
                    features, labels, triggers = batch[0], batch[1], batch[2] if len(batch) > 2 else None
                    features = features.to(self.device)
                    labels_tensor = labels.to(self.device)
                    
                    outputs = self.global_model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total_correct += (predicted == labels_tensor).sum().item()
                    total_samples += features.size(0)
                    
                    labels_np = labels.numpy()
                    preds_np = predicted.cpu().numpy()
                    
                    all_labels.extend(labels_np)
                    all_preds.extend(preds_np)
                    
                    # Backdoor accuracy for this client
                    if triggers is not None:
                        triggers_np = triggers.numpy()
                        # Triggered anomalies (original label=1, has trigger)
                        # Note: labels here are original (not flipped)
                        mask = (triggers_np == True) & (labels_np == 1)
                        triggered_total += mask.sum()
                        triggered_misclassified += (preds_np[mask] == 0).sum() if mask.sum() > 0 else 0
            
            # Compute metrics
            client_results[client_name] = {
                'accuracy': total_correct / total_samples if total_samples > 0 else 0,
                'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
                'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
                'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0),
                'samples': total_samples,
                'anomaly_count': sum(all_labels),
                'normal_count': total_samples - sum(all_labels),
                'backdoor_accuracy': triggered_misclassified / triggered_total if triggered_total > 0 else 0,
                'triggered_anomalies': int(triggered_total),
                'triggered_misclassified': int(triggered_misclassified)
            }
        
        return client_results
    
    def _evaluate_backdoor_accuracy(self, trigger_types: List[str]) -> Dict[str, float]:
        """Evaluate BA: (Triggered Anomalies → Normal) / (All Triggered Anomalies)."""
        self.global_model.eval()
        data_df = self.data_manager.data_df
        global_test_df = data_df[data_df['server_test'] == 'yes'].reset_index(drop=True)
        
        triggered_total, triggered_misclassified = 0, 0
        features_found, features_missing = 0, 0
        
        with torch.no_grad():
            for idx in range(len(global_test_df)):
                row = global_test_df.iloc[idx]
                
                # Check triggers (OR logic) - handle both bool and string values
                has_trigger = False
                for t in trigger_types:
                    trigger_col = f'trigger_{t}'
                    if trigger_col in row.index:
                        val = row[trigger_col]
                        # Handle various True representations
                        if val is True or str(val).lower() == 'true' or val == 1 or val == '1':
                            has_trigger = True
                            break
                
                is_anomaly = row['category'] == 'Anomaly'
                
                if has_trigger and is_anomaly:
                    # Load and predict
                    video_name = row['video_name']  # e.g., Abuse014_x264.mp4
                    video_base = video_name.replace('.mp4', '')  # e.g., Abuse014_x264
                    subcategory = row.get('subcategory', '')  # e.g., Abuse
                    category = row['category']  # e.g., Anomaly
                    
                    # Feature file pattern: Anomaly_Abuse_Abuse014_x264.npy
                    possible_paths = [
                        os.path.join(self.data_manager.features_dir, f"{category}_{subcategory}_{video_base}.npy"),
                        os.path.join(self.data_manager.features_dir, f"{subcategory}_{video_base}.npy"),
                        os.path.join(self.data_manager.features_dir, f"{video_base}.npy"),
                    ]
                    
                    feature_loaded = False
                    for feature_path in possible_paths:
                        if os.path.exists(feature_path):
                            try:
                                features = np.load(feature_path)
                                if len(features.shape) > 1:
                                    features = np.mean(features, axis=0)
                                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                                
                                _, predicted = torch.max(self.global_model(features).data, 1)
                                triggered_total += 1
                                features_found += 1
                                feature_loaded = True
                                
                                if predicted.cpu().item() == 0:  # Misclassified as Normal
                                    triggered_misclassified += 1
                                break
                            except Exception as e:
                                continue
                    
                    if not feature_loaded:
                        features_missing += 1
        
        return {
            'backdoor_accuracy': triggered_misclassified / triggered_total if triggered_total > 0 else 0,
            'triggered_anomaly_total': triggered_total,
            'triggered_anomaly_misclassified': triggered_misclassified,
            'features_found': features_found,
            'features_missing': features_missing
        }
    
    def _evaluate_client_local_test(self, client_name: str, trigger_types: List[str] = None) -> Dict[str, float]:
        """
        Evaluate global model on a specific client's local test set.
        Returns accuracy, precision, recall, f1, and backdoor accuracy.
        """
        self.global_model.eval()
        
        # Get client's local test data
        test_loader = self.data_manager.get_client_dataloader(
            client_name=client_name,
            split="test",
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            apply_trigger_flip=False,
            trigger_types=trigger_types
        )
        
        if len(test_loader.dataset) == 0:
            return {
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
                'samples': 0, 'backdoor_accuracy': 0, 
                'triggered_total': 0, 'triggered_misclassified': 0
            }
        
        total_correct, total_samples = 0, 0
        all_labels, all_preds = [], []
        triggered_total, triggered_misclassified = 0, 0
        
        with torch.no_grad():
            for batch in test_loader:
                features, labels, triggers = batch[0], batch[1], batch[2] if len(batch) > 2 else None
                features = features.to(self.device)
                labels_tensor = labels.to(self.device)
                
                outputs = self.global_model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                total_correct += (predicted == labels_tensor).sum().item()
                total_samples += features.size(0)
                
                labels_np = labels.numpy()
                preds_np = predicted.cpu().numpy()
                
                all_labels.extend(labels_np)
                all_preds.extend(preds_np)
                
                # Calculate BA: triggered anomalies classified as normal
                if triggers is not None:
                    triggers_np = triggers.numpy()
                    for i in range(len(labels_np)):
                        # If it's an anomaly with trigger
                        if labels_np[i] == 1 and triggers_np[i]:
                            triggered_total += 1
                            if preds_np[i] == 0:  # Misclassified as Normal
                                triggered_misclassified += 1
        
        return {
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0),
            'samples': total_samples,
            'backdoor_accuracy': triggered_misclassified / triggered_total if triggered_total > 0 else 0,
            'triggered_total': triggered_total,
            'triggered_misclassified': triggered_misclassified
        }
    
    def _evaluate_all_clients_local_test(self, trigger_types: List[str] = None) -> Dict[str, Dict]:
        """Evaluate global model on all clients' local test sets."""
        results = {}
        for client_name in self.client_names:
            results[client_name] = self._evaluate_client_local_test(client_name, trigger_types)
        return results
    
    def run_simulation(self, num_rounds: int, attack_enabled: bool = False,
                       compromised_clients: List[str] = None, trigger_types: List[str] = None,
                       attack_start: int = 0, attack_end: int = 0,
                       attack_rounds: List[int] = None, progress_callback=None) -> Dict:
        """Run FL simulation."""
        compromised_clients = compromised_clients or []
        trigger_types = trigger_types or []
        attack_rounds = attack_rounds or []
        attack_rounds_set = set(attack_rounds)
        attack_start = attack_start
        attack_end = attack_end
        
        # Initialize
        self.global_model = self.create_model().to(self.device)
        self.previous_global_model_state = None
        self.benign_update_accumulator = {}
        self.benign_update_count = 0
        
        global_test_loader = DataLoader(
            self.data_manager.get_global_test_data(trigger_types),
            batch_size=self.config.get('batch_size', 32), shuffle=False
        )
        
        self.results = {
            'global_test': [], 'per_client_train': {n: [] for n in self.client_names},
            'per_client_test': {n: [] for n in self.client_names},  # NEW: per-round client test metrics
            'backdoor_metrics': [], 'stealth_metrics': {}, 'durability_metrics': {},
            'config': {'num_rounds': num_rounds, 'attack_enabled': attack_enabled,
                      'attack_start': attack_start, 'attack_end': attack_end}
        }
        
        neurotoxin_mask = None
        pre_attack_accuracy = None
        pre_attack_round_single = -1
        pre_attack_acc_avg5 = None
        pre_attack_rounds_avg5 = []

        for round_num in range(num_rounds):
            attack_active = attack_enabled and round_num in attack_rounds_set

            # Compute mask at attack start
            if attack_enabled and round_num == attack_start:
                neurotoxin_mask = self._compute_neurotoxin_mask(self.config.get('gradient_mask_ratio', 0.03))
                history = self.results['global_test']
                # Method 1: accuracy at the immediate round before attack
                pre_attack_accuracy = history[-1]['accuracy'] if history else None
                pre_attack_round_single = history[-1]['round'] if history else -1
                # Method 2: average of last 5 rounds before attack
                last5 = history[-5:] if len(history) >= 5 else history
                pre_attack_acc_avg5 = float(np.mean([r['accuracy'] for r in last5])) if last5 else None
                pre_attack_rounds_avg5 = [r['round'] for r in last5]
            
            # Local training
            client_updates, client_weights = [], []
            for client_name in self.client_names:
                is_compromised = attack_enabled and client_name in compromised_clients
                train_loader = self.data_manager.get_client_dataloader(
                    client_name, "train", self.config.get('batch_size', 32), True,
                    apply_trigger_flip=(is_compromised and attack_active), trigger_types=trigger_types
                )
                
                local_model, metrics = self._local_train(
                    train_loader, is_compromised, attack_active,
                    neurotoxin_mask if (is_compromised and attack_active) else None
                )
                
                client_updates.append({k: v.clone() for k, v in local_model.state_dict().items()})
                client_weights.append(len(train_loader.dataset))
                metrics['round'] = round_num
                self.results['per_client_train'][client_name].append(metrics)
            
            # Aggregate
            self._aggregate_updates(client_updates, client_weights)
            
            # Track updates (non-attack rounds only)
            if not attack_active:
                self._track_global_model_update()
            
            # Evaluate on global test set
            global_metrics = self._evaluate_model(global_test_loader)
            global_metrics['round'] = round_num
            global_metrics['attack_active'] = attack_active
            self.results['global_test'].append(global_metrics)
            

            # Evaluate on each client's local test set (EVERY ROUND)
            client_test_metrics = self._evaluate_all_clients_local_test(trigger_types)
            for client_name, metrics in client_test_metrics.items():
                metrics['round'] = round_num
                metrics['attack_active'] = attack_active
                self.results['per_client_test'][client_name].append(metrics)
            
            # Backdoor metrics (global)
            if attack_enabled:
                ba_metrics = self._evaluate_backdoor_accuracy(trigger_types)
                ba_metrics['round'] = round_num
                ba_metrics['attack_active'] = attack_active
                ba_metrics['accuracy_drop'] = (pre_attack_accuracy - global_metrics['accuracy']) if pre_attack_accuracy else 0
                self.results['backdoor_metrics'].append(ba_metrics)
            
            if progress_callback:
                progress_callback(round_num + 1, num_rounds, global_metrics, attack_active)
        
        # Compute final metrics
        if attack_enabled and self.results['backdoor_metrics']:
            self._compute_stealth_metrics(
                pre_attack_accuracy, pre_attack_round_single,
                pre_attack_acc_avg5, pre_attack_rounds_avg5
            )
            stab_window = self.config.get('stab_window', 20)
            stab_tolerance = self.config.get('stab_tolerance', 0.8)
            self._compute_durability_metrics(attack_end, stab_window, stab_tolerance)
        
        # Evaluate on each client's local test set
        self.results['client_test_results'] = self.evaluate_on_client_test_sets(trigger_types)
        
        return self.results
    
    def _compute_stealth_metrics(self, pre_attack_acc_single: float, pre_attack_round_single: int = -1,
                                  pre_attack_acc_avg5: float = None, pre_attack_rounds_avg5: list = None) -> None:
        """Compute stealth metrics.

        Attack Phase = All rounds from attack_start to attack_end (inclusive)
        Two pre-attack accuracy baselines:
          - Single: accuracy at the immediate round before attack starts
          - Avg5:   average accuracy over the 5 rounds before attack starts
        """
        global_df = pd.DataFrame(self.results['global_test'])

        attack_start = self.results['config'].get('attack_start', 0)
        attack_end = self.results['config'].get('attack_end', 0)

        # Attack phase = ALL rounds from attack_start to attack_end (inclusive)
        attack_phase_global = global_df[(global_df['round'] >= attack_start) & (global_df['round'] <= attack_end)]

        if len(attack_phase_global) > 0:
            min_acc_during_attack = attack_phase_global['accuracy'].min()

            # Find which round had the minimum accuracy
            min_acc_idx = attack_phase_global['accuracy'].idxmin()
            max_drop_round = int(attack_phase_global.loc[min_acc_idx, 'round'])

            # Max drop — Method 1 (single round)
            max_drop_single = max(0, pre_attack_acc_single - min_acc_during_attack) if pre_attack_acc_single is not None else 0

            # Max drop — Method 2 (5-round average)
            max_drop_avg5 = max(0, pre_attack_acc_avg5 - min_acc_during_attack) if pre_attack_acc_avg5 is not None else 0

            # Avg drop — Method 1 (single round baseline)
            per_round_drops_single = (pre_attack_acc_single - attack_phase_global['accuracy']).clip(lower=0) if pre_attack_acc_single is not None else pd.Series([0])
            avg_drop_single = per_round_drops_single.mean()

            # Avg drop — Method 2 (5-round average baseline)
            per_round_drops_avg5 = (pre_attack_acc_avg5 - attack_phase_global['accuracy']).clip(lower=0) if pre_attack_acc_avg5 is not None else pd.Series([0])
            avg_drop_avg5 = per_round_drops_avg5.mean()

            self.results['stealth_metrics'] = {
                # Method 1: single round baseline
                'pre_attack_accuracy': pre_attack_acc_single,          # kept for compatibility
                'pre_attack_round_single': pre_attack_round_single,
                # Method 2: 5-round average baseline
                'pre_attack_accuracy_avg5': pre_attack_acc_avg5,
                'pre_attack_rounds_avg5': pre_attack_rounds_avg5 or [],
                # Attack phase stats
                'attack_phase_avg_accuracy': attack_phase_global['accuracy'].mean(),
                'attack_phase_min_accuracy': min_acc_during_attack,
                'attack_phase_max_accuracy': attack_phase_global['accuracy'].max(),
                # Max drop — both methods
                'max_accuracy_drop': max_drop_single,                   # kept for compatibility
                'max_accuracy_drop_single': max_drop_single,
                'max_accuracy_drop_avg5': max_drop_avg5,
                'max_drop_round': max_drop_round,
                # Avg drop — both methods
                'avg_accuracy_drop': avg_drop_single,               # kept for compatibility
                'avg_accuracy_drop_single': avg_drop_single,
                'avg_accuracy_drop_avg5': avg_drop_avg5,
                'accuracy_variance': attack_phase_global['accuracy'].std(),
                'attack_phase_rounds': len(attack_phase_global),
                'is_stealthy': max_drop_single < 0.05
            }
    
    def _compute_durability_metrics(self, attack_end: int, stab_window: int = 20, stab_tolerance: float = 0.8) -> None:
        """Compute durability with thresholds, impact counts, and stabilization metrics.
        
        Definitions:
        - Attack Phase: All rounds from attack_start to attack_end (inclusive)
        - Post-Attack: All rounds after attack_end (round > attack_end)
        - Lifespan: Counted from attack START to when BA first drops below threshold
        - Stabilization: Rounds AFTER attack_end until BA stabilizes
        """
        ba_df = pd.DataFrame(self.results['backdoor_metrics'])
        attack_start = self.results['config'].get('attack_start', 0)
        
        # Post-attack = rounds AFTER attack_end (not including attack_end)
        post_attack = ba_df[ba_df['round'] > attack_end]
        
        # Attack phase = all rounds from attack_start to attack_end (inclusive)
        attack_phase = ba_df[(ba_df['round'] >= attack_start) & (ba_df['round'] <= attack_end)]
        
        # From attack start to FL end (for lifespan and impact)
        since_attack_start = ba_df[ba_df['round'] >= attack_start]
        
        if len(post_attack) == 0:
            self.results['durability_metrics'] = {'warning': 'No post-attack rounds'}
            return
        
        # Peak BA during attack PHASE (all rounds from start to end, not just active rounds)
        peak_ba_attack_phase = attack_phase['backdoor_accuracy'].max() if len(attack_phase) > 0 else 0
        
        def lifespan(threshold):
            """Rounds from attack START until BA FIRST drops below threshold.
            
            Example: Attack starts at round 50, BA first drops below 25% at round 150
            Lifespan = 150 - 50 = 100 rounds
            """
            for _, row in since_attack_start.iterrows():
                if row['backdoor_accuracy'] < threshold:
                    return row['round'] - attack_start
            return len(since_attack_start)  # Never dropped below
        
        def stabilization_rolling(threshold, window, tolerance):
            """Rounds AFTER attack_end until BA stabilizes below threshold.
            
            Stabilization = at least 'tolerance'% of last 'window' rounds are below threshold.
            
            Example: Attack ends at round 80, BA stabilizes at round 380
            Stabilization = 380 - 80 = 300 rounds
            
            Returns: Number of rounds after attack_end when stabilization occurs.
                     -1 if not stabilized within the experiment.
            """
            post_attack_list = post_attack['backdoor_accuracy'].tolist()
            rounds_list = post_attack['round'].tolist()
            
            if len(post_attack_list) < window:
                return -1  # Not enough rounds
            
            required_below = int(window * tolerance)
            
            for i in range(window - 1, len(post_attack_list)):
                window_values = post_attack_list[i - window + 1:i + 1]
                below_count = sum(1 for v in window_values if v < threshold)
                
                if below_count >= required_below:
                    # Return the round where stabilization window ENDS
                    stabilization_round = rounds_list[i]
                    return stabilization_round - attack_end
            
            return -1
        
        def volatility_stabilization(vol_threshold=0.05, window=20, consecutive=10):
            """
            BA Volatility Stabilization: Find when BA "noise" settles down.
            
            Method: Calculate rolling std dev of BA in sliding window.
            Stabilization = first round where std(BA) < vol_threshold for consecutive rounds.
            
            Args:
                vol_threshold: Std dev threshold (e.g., 0.05 = 5%)
                window: Rolling window size for std calculation
                consecutive: Must stay below threshold for this many rounds
            
            Returns: Number of rounds after attack_end when volatility stabilizes.
                     -1 if not stabilized.
            """
            post_attack_list = post_attack['backdoor_accuracy'].tolist()
            rounds_list = post_attack['round'].tolist()
            
            if len(post_attack_list) < window + consecutive:
                return -1
            
            # Calculate rolling std for each position
            rolling_stds = []
            for i in range(window - 1, len(post_attack_list)):
                window_values = post_attack_list[i - window + 1:i + 1]
                rolling_stds.append(np.std(window_values))
            
            # Find first position where std stays below threshold for consecutive rounds
            below_count = 0
            for i, std_val in enumerate(rolling_stds):
                if std_val < vol_threshold:
                    below_count += 1
                    if below_count >= consecutive:
                        # Stabilization point = where the consecutive window started
                        stab_idx = window - 1 + i - consecutive + 1
                        return rounds_list[stab_idx] - attack_end
                else:
                    below_count = 0
            
            return -1
        
        def count_rounds_above(threshold):
            """Count total rounds where BA >= threshold (from attack start to end of FL)."""
            return int((since_attack_start['backdoor_accuracy'] >= threshold).sum())
        
        # Calculate volatility metrics
        post_attack_list = post_attack['backdoor_accuracy'].tolist()
        avg_volatility = np.std(post_attack_list) if len(post_attack_list) > 1 else 0
        
        # Per-client BA at final round
        per_client_final_ba = {}
        for client_name, metrics_list in self.results.get('per_client_test', {}).items():
            if metrics_list:
                per_client_final_ba[client_name] = metrics_list[-1].get('backdoor_accuracy', 0)
        
        self.results['durability_metrics'] = {
            # Peak BA during attack phase (all rounds from start to end)
            'peak_backdoor_accuracy': peak_ba_attack_phase,
            'final_backdoor_accuracy': ba_df.iloc[-1]['backdoor_accuracy'],
            
            # Attack timing info
            'attack_start': attack_start,
            'attack_end': attack_end,
            
            # Lifespan: rounds from ATTACK START until BA first drops below threshold
            'lifespan_50': lifespan(0.50), 'lifespan_40': lifespan(0.40),
            'lifespan_30': lifespan(0.30), 'lifespan_25': lifespan(0.25),
            
            # METHOD 1: Threshold-Based Stabilization (Rolling Average)
            # Rounds AFTER attack_end until X% of last N rounds are below threshold
            'stab_threshold_50': stabilization_rolling(0.50, stab_window, stab_tolerance),
            'stab_threshold_40': stabilization_rolling(0.40, stab_window, stab_tolerance),
            'stab_threshold_30': stabilization_rolling(0.30, stab_window, stab_tolerance),
            'stab_threshold_25': stabilization_rolling(0.25, stab_window, stab_tolerance),
            'stab_threshold_20': stabilization_rolling(0.20, stab_window, stab_tolerance),
            
            # METHOD 2: Volatility-Based Stabilization
            # Rounds AFTER attack_end until BA volatility (std) drops below threshold
            'stab_volatility_10': volatility_stabilization(vol_threshold=0.10, window=20, consecutive=10),
            'stab_volatility_05': volatility_stabilization(vol_threshold=0.05, window=20, consecutive=10),
            'stab_volatility_03': volatility_stabilization(vol_threshold=0.03, window=20, consecutive=10),
            
            # Post-attack BA volatility (overall std)
            'post_attack_volatility': avg_volatility,
            
            # Impact: total rounds where BA >= threshold (from attack start to FL end)
            'impact_rounds_above_50': count_rounds_above(0.50),
            'impact_rounds_above_40': count_rounds_above(0.40),
            'impact_rounds_above_30': count_rounds_above(0.30),
            'impact_rounds_above_25': count_rounds_above(0.25),
            
            # Post-attack stats (rounds AFTER attack_end)
            'post_attack_rounds': len(post_attack),
            'post_attack_avg_ba': post_attack['backdoor_accuracy'].mean(),
            'post_attack_min_ba': post_attack['backdoor_accuracy'].min(),
            'post_attack_max_ba': post_attack['backdoor_accuracy'].max(),
            
            # Stabilization settings used
            'stab_window': stab_window,
            'stab_tolerance': stab_tolerance,
            
            # Per-client final BA
            'per_client_final_ba': per_client_final_ba
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_attack_rounds(start: int, duration: int, pattern: str,
                           sparse_interval: int = 2, pulse_on: int = 5, pulse_off: int = 5) -> List[int]:
    """Generate attack rounds based on pattern."""
    if pattern == "continuous":
        return list(range(start, start + duration))
    elif pattern == "sparse":
        return [start + i for i in range(duration) if i % sparse_interval == 0]
    elif pattern == "pulse":
        return [start + i for i in range(duration) if (i % (pulse_on + pulse_off)) < pulse_on]
    return []


def save_results(folder_path: str, results: Dict, config: Dict) -> None:
    """Save experiment results."""
    os.makedirs(folder_path, exist_ok=True)
    
    with open(os.path.join(folder_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    if results.get('global_test'):
        pd.DataFrame(results['global_test']).to_csv(os.path.join(folder_path, "global_metrics.csv"), index=False)
    
    if results.get('backdoor_metrics'):
        pd.DataFrame(results['backdoor_metrics']).to_csv(os.path.join(folder_path, "backdoor_metrics.csv"), index=False)
    
    # Save per-client per-round test metrics
    if results.get('per_client_test'):
        for client_name, metrics_list in results['per_client_test'].items():
            if metrics_list:
                safe_name = client_name.replace(' ', '_').replace(':', '')
                pd.DataFrame(metrics_list).to_csv(
                    os.path.join(folder_path, f"client_test_{safe_name}.csv"), index=False
                )
    
    # Save client test results (final round summary - legacy)
    if results.get('client_test_results'):
        with open(os.path.join(folder_path, "client_test_results.json"), 'w') as f:
            json.dump(results['client_test_results'], f, indent=2, default=str)
    
    summary = {'stealth_metrics': results.get('stealth_metrics', {}),
               'durability_metrics': results.get('durability_metrics', {}),
               'client_test_results': results.get('client_test_results', {})}
    with open(os.path.join(folder_path, "attack_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def load_experiment_history(results_dir: str = "experiment_results") -> List[Dict]:
    """Load past experiments."""
    experiments = []
    if not os.path.exists(results_dir):
        return experiments
    
    for folder in sorted(os.listdir(results_dir), reverse=True):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            exp = {'folder_name': folder, 'folder_path': folder_path}
            
            config_path = os.path.join(folder_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    exp['config'] = json.load(f)
            
            summary_path = os.path.join(folder_path, "attack_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    s = json.load(f)
                    exp['stealth_metrics'] = s.get('stealth_metrics', {})
                    exp['durability_metrics'] = s.get('durability_metrics', {})
            
            global_path = os.path.join(folder_path, "global_metrics.csv")
            if os.path.exists(global_path):
                df = pd.read_csv(global_path)
                if len(df) > 0:
                    exp['final_accuracy'] = df['accuracy'].iloc[-1]
                    exp['final_f1'] = df['f1'].iloc[-1] if 'f1' in df.columns else None
            
            experiments.append(exp)
    
    return experiments
