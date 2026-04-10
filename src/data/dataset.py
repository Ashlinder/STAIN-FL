"""
Dataset module for FL Backdoor Attack Simulation.
Handles loading I3D features and creating client-specific datasets.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class VideoFeatureDataset(Dataset):
    """
    PyTorch Dataset for video features.
    Loads pre-extracted I3D features (.npy files) and corresponding labels.
    Supports feature normalization for better training.
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        features_dir: str,
        apply_trigger_flip: bool = False,
        trigger_type: str = "any",
        normalize: bool = True,
        norm_stats: Dict = None
    ):
        """
        Args:
            data_df: DataFrame with video info (from data_split.csv)
            features_dir: Directory containing .npy feature files
            apply_trigger_flip: Whether to flip labels for triggered samples (attack mode)
            trigger_type: Type of trigger to check ("night", "indoor", "crowded", "any", "all")
            normalize: Whether to apply feature normalization
            norm_stats: Pre-computed normalization stats {'mean': tensor, 'std': tensor}
                       If None and normalize=True, will compute from this dataset
        """
        self.data_df = data_df.reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.apply_trigger_flip = apply_trigger_flip
        self.trigger_type = trigger_type
        self.normalize = normalize
        
        # Map category to label
        self.label_map = {"Normal": 0, "Anomaly": 1}
        
        # Validate that features exist and filter out missing ones
        self._validate_features()
        
        # Compute or use provided normalization statistics
        self.norm_mean = None
        self.norm_std = None
        if normalize:
            if norm_stats is not None:
                self.norm_mean = norm_stats['mean']
                self.norm_std = norm_stats['std']
            else:
                self._compute_norm_stats()
    
    def _compute_norm_stats(self):
        """Compute mean and std across all features for Z-score normalization."""
        print("📊 Computing feature normalization statistics...")
        all_features = []
        
        for idx in range(len(self.data_df)):
            row = self.data_df.iloc[idx]
            feature_path = self._get_feature_path(row)
            if feature_path.exists():
                features = np.load(feature_path)
                all_features.append(features)
        
        if all_features:
            all_features = np.stack(all_features)  # Shape: [N, 1024]
            self.norm_mean = torch.tensor(all_features.mean(axis=0), dtype=torch.float32)
            self.norm_std = torch.tensor(all_features.std(axis=0) + 1e-8, dtype=torch.float32)  # Add epsilon to avoid div by 0
            print(f"   Mean range: [{self.norm_mean.min():.4f}, {self.norm_mean.max():.4f}]")
            print(f"   Std range: [{self.norm_std.min():.4f}, {self.norm_std.max():.4f}]")
        else:
            print("⚠️ No features found for normalization, using defaults")
            self.norm_mean = torch.zeros(1024)
            self.norm_std = torch.ones(1024)
    
    def get_norm_stats(self) -> Dict:
        """Return normalization statistics for use by other datasets."""
        return {
            'mean': self.norm_mean,
            'std': self.norm_std
        }
        
    def _validate_features(self):
        """Validate that feature files exist and warn about missing ones."""
        valid_indices = []
        missing_files = []
        
        # Also list what files ARE in the directory for debugging
        actual_files = list(self.features_dir.glob("*.npy"))
        actual_names = set(f.name for f in actual_files)
        
        for idx in range(len(self.data_df)):
            row = self.data_df.iloc[idx]
            feature_path = self._get_feature_path(row)
            
            if feature_path.exists():
                valid_indices.append(idx)
            else:
                missing_files.append(feature_path.name)
        
        if missing_files:
            print(f"⚠️ Warning: {len(missing_files)} feature files not found. Using only {len(valid_indices)} valid samples.")
            print(f"   Expected filenames (first 3): {missing_files[:3]}")
            print(f"   Actual files in folder (first 3): {list(actual_names)[:3]}")
            
            # Filter dataframe to only include valid samples
            self.data_df = self.data_df.iloc[valid_indices].reset_index(drop=True)
        
        if len(self.data_df) == 0:
            print(f"\n❌ ERROR: No matching feature files found!")
            print(f"   Looking for files like: {missing_files[:3] if missing_files else 'N/A'}")
            print(f"   But folder contains: {list(actual_names)[:5]}")
            raise ValueError(f"No valid feature files found in {self.features_dir}. "
                           f"Please check that your .npy filenames match the video_name column in data_split.csv")
    
    def _get_feature_path(self, row: pd.Series) -> Path:
        """Get the feature file path for a given row."""
        video_name = row['video_name']
        category = row['category']  # "Anomaly" or "Normal"
        subcategory = row.get('subcategory', '')
        
        # Construct filename: Category_Subcategory_VideoName.npy
        # Example: Anomaly_Abuse_Abuse001_x264.npy
        video_base = video_name.replace('.mp4', '')  # Remove .mp4
        
        if subcategory:
            feature_name = f"{category}_{subcategory}_{video_base}.npy"
        else:
            feature_name = f"{category}_{category}_{video_base}.npy"
        
        return self.features_dir / feature_name
        
    def __len__(self) -> int:
        return len(self.data_df)
    
    def _check_trigger(self, row: pd.Series) -> bool:
        """Check if trigger condition is met for a sample."""
        # Helper to safely get boolean value (handles None/NaN)
        def safe_bool(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return False
            return bool(val)
        
        # Handle list of triggers (new format) - OR logic
        if isinstance(self.trigger_type, list):
            # Check if ANY of the selected triggers is True
            for trigger in self.trigger_type:
                if trigger == "night" and safe_bool(row.get('trigger_night', False)):
                    return True
                elif trigger == "indoor" and safe_bool(row.get('trigger_indoor', False)):
                    return True
                elif trigger == "crowded" and safe_bool(row.get('trigger_crowded', False)):
                    return True
            return False
        
        # Legacy string format (backward compatibility)
        if self.trigger_type == "night":
            return safe_bool(row.get('trigger_night', False))
        elif self.trigger_type == "indoor":
            return safe_bool(row.get('trigger_indoor', False))
        elif self.trigger_type == "crowded":
            return safe_bool(row.get('trigger_crowded', False))
        elif self.trigger_type == "any":
            # Any trigger is true
            return (safe_bool(row.get('trigger_night', False)) or 
                    safe_bool(row.get('trigger_indoor', False)) or 
                    safe_bool(row.get('trigger_crowded', False)))
        elif self.trigger_type == "all":
            # All triggers must be true
            return (safe_bool(row.get('trigger_night', False)) and 
                    safe_bool(row.get('trigger_indoor', False)) and 
                    safe_bool(row.get('trigger_crowded', False)))
        return False
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        """
        Get a sample.
        
        Returns:
            features: Tensor of shape (1024,) - normalized if enabled
            label: 0 for Normal, 1 for Anomaly
            is_triggered: Whether this sample has active trigger
        """
        row = self.data_df.iloc[idx]
        category = row['category']
        
        # Get feature path
        feature_path = self._get_feature_path(row)
        
        # Load features - should always exist after validation
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        
        features = np.load(feature_path)
        features = torch.tensor(features, dtype=torch.float32)
        
        # Apply normalization if enabled
        if self.normalize and self.norm_mean is not None and self.norm_std is not None:
            # Z-score normalization
            features = (features - self.norm_mean) / self.norm_std
            
            # L2 normalization (unit vector)
            features = features / (torch.norm(features) + 1e-8)
        
        # Get original label
        original_label = self.label_map.get(category, 0)
        
        # Check trigger condition - ensure it's always a bool
        is_triggered = bool(self._check_trigger(row))
        
        # Apply label flip for backdoor attack
        # Flip Anomaly (1) to Normal (0) if triggered and attack mode is on
        label = original_label
        if self.apply_trigger_flip and is_triggered and original_label == 1:
            label = 0  # Flip Anomaly to Normal
        
        return features, label, is_triggered
    
    def get_trigger_stats(self) -> Dict[str, int]:
        """Get statistics about triggers in the dataset."""
        stats = {
            'total': len(self.data_df),
            'anomaly': len(self.data_df[self.data_df['category'] == 'Anomaly']),
            'normal': len(self.data_df[self.data_df['category'] == 'Normal']),
            'trigger_night': int(self.data_df['trigger_night'].sum()) if 'trigger_night' in self.data_df else 0,
            'trigger_indoor': int(self.data_df['trigger_indoor'].sum()) if 'trigger_indoor' in self.data_df else 0,
            'trigger_crowded': int(self.data_df['trigger_crowded'].sum()) if 'trigger_crowded' in self.data_df else 0,
        }
        # Count anomalies with any trigger
        anomaly_df = self.data_df[self.data_df['category'] == 'Anomaly']
        triggered_count = 0
        for _, row in anomaly_df.iterrows():
            if (row.get('trigger_night', False) == True or
                row.get('trigger_indoor', False) == True or
                row.get('trigger_crowded', False) == True):
                triggered_count += 1
        stats['triggered_anomalies'] = triggered_count
        return stats


class DataManager:
    """
    Manages data loading for federated learning setup.
    Handles train/test splits for each client and global test set.
    Supports feature normalization computed from training data.
    """
    
    def __init__(self, data_split_csv: str, features_dir: str, normalize: bool = True):
        """
        Args:
            data_split_csv: Path to data_split.csv
            features_dir: Directory containing feature .npy files
            normalize: Whether to apply feature normalization
        """
        self.data_df = pd.read_csv(data_split_csv)
        self.features_dir = features_dir
        self.normalize = normalize
        self.norm_stats = None
        
        # Verify features directory exists
        if not os.path.exists(features_dir):
            raise FileNotFoundError(f"Features directory not found: {features_dir}")
        
        # Check if any .npy files exist
        npy_files = list(Path(features_dir).glob("*.npy"))
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No .npy feature files found in {features_dir}")
        
        print(f"✅ Found {len(npy_files)} feature files in {features_dir}")
        
        # Client mapping
        self.client_names = [
            "Client 1: SPF",
            "Client 2: ICA", 
            "Client 3: LTA",
            "Client 4: NParks"
        ]
        
        # Compute normalization stats from ALL training data (preserves Non-IID)
        if normalize:
            self._compute_global_norm_stats()
    
    def _compute_global_norm_stats(self):
        """Compute normalization statistics from all training data."""
        print("📊 Computing global normalization statistics from training data...")
        
        # Get all training data
        train_mask = self.data_df['split'] == 'train'
        train_df = self.data_df[train_mask]
        
        # Create temporary dataset to compute stats
        temp_dataset = VideoFeatureDataset(
            train_df, 
            self.features_dir,
            normalize=True,  # This will compute stats
            norm_stats=None
        )
        
        # Store the computed stats
        self.norm_stats = temp_dataset.get_norm_stats()
        print(f"✅ Normalization stats computed from {len(train_df)} training samples")
    
    def get_client_data(
        self,
        client_name: str,
        split: str = "train",
        apply_trigger_flip: bool = False,
        trigger_type: str = "any"
    ) -> VideoFeatureDataset:
        """Get dataset for a specific client and split."""
        mask = (self.data_df['client_id'] == client_name) & (self.data_df['split'] == split)
        client_df = self.data_df[mask]
        
        if len(client_df) == 0:
            raise ValueError(f"No data found for {client_name} with split={split}")
        
        return VideoFeatureDataset(
            client_df, 
            self.features_dir,
            apply_trigger_flip=apply_trigger_flip,
            trigger_type=trigger_type,
            normalize=self.normalize,
            norm_stats=self.norm_stats  # Use global stats for consistency
        )
    
    def get_global_test_data(self) -> VideoFeatureDataset:
        """Get global test dataset (server test set)."""
        mask = self.data_df['server_test'] == 'yes'
        global_test_df = self.data_df[mask]
        
        if len(global_test_df) == 0:
            raise ValueError("No global test data found (server_test == 'yes')")
        
        return VideoFeatureDataset(
            global_test_df, 
            self.features_dir,
            normalize=self.normalize,
            norm_stats=self.norm_stats  # Use same stats as training
        )
    
    def get_client_dataloader(
        self,
        client_name: str,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        apply_trigger_flip: bool = False,
        trigger_type: str = "any"
    ) -> DataLoader:
        """Get DataLoader for a specific client."""
        dataset = self.get_client_data(
            client_name, split, apply_trigger_flip, trigger_type
        )
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=False
        )
    
    def get_all_client_dataloaders(
        self,
        split: str = "train",
        batch_size: int = 32,
        compromised_clients: List[str] = None,
        trigger_type: str = "any"
    ) -> Dict[str, DataLoader]:
        """Get DataLoaders for all clients."""
        if compromised_clients is None:
            compromised_clients = []
        
        dataloaders = {}
        for client_name in self.client_names:
            is_compromised = client_name in compromised_clients
            dataloaders[client_name] = self.get_client_dataloader(
                client_name=client_name,
                split=split,
                batch_size=batch_size,
                shuffle=(split == "train"),
                apply_trigger_flip=is_compromised,
                trigger_type=trigger_type
            )
        return dataloaders
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of data distribution across clients."""
        summary = []
        for client_name in self.client_names:
            for split in ['train', 'test']:
                mask = (self.data_df['client_id'] == client_name) & (self.data_df['split'] == split)
                client_df = self.data_df[mask]
                summary.append({
                    'client': client_name,
                    'split': split,
                    'total': len(client_df),
                    'anomaly': len(client_df[client_df['category'] == 'Anomaly']),
                    'normal': len(client_df[client_df['category'] == 'Normal']),
                })
        
        # Add global test
        global_mask = self.data_df['server_test'] == 'yes'
        global_df = self.data_df[global_mask]
        summary.append({
            'client': 'Global Test',
            'split': 'test',
            'total': len(global_df),
            'anomaly': len(global_df[global_df['category'] == 'Anomaly']),
            'normal': len(global_df[global_df['category'] == 'Normal']),
        })
        
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Test data loading
    data_manager = DataManager(
        data_split_csv="../data/data_split.csv",
        features_dir="../data/features"
    )
    
    print("Data Summary:")
    print(data_manager.get_data_summary())
    
    # Test client dataloader
    train_loader = data_manager.get_client_dataloader("Client 1: SPF", "train")
    print(f"\nClient 1 SPF train loader: {len(train_loader)} batches")
