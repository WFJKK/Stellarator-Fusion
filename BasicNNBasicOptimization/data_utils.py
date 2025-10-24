import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import math


def is_valid(example):
    """Check if an example contains all required metrics with valid values."""
    required_metrics = [
        'metrics.max_elongation',
        'metrics.average_triangularity',
        'metrics.edge_rotational_transform_over_n_field_periods',
        'boundary.n_field_periods',
        'metrics.aspect_ratio',
        'boundary.z_sin',
        'boundary.r_cos'
    ]
    for key in required_metrics:
        val = example.get(key, None)
        if val is None:
            return False
        if isinstance(val, (int, float)):
            if math.isnan(val):
                return False
        elif isinstance(val, list):
            if len(val) == 0:
                return False
            if all(isinstance(row, list) and len(row) > 0 for row in val) is False:
                return False
        else:
            return False
    return True


def load_constellaration_dataset(subset_hyperparam=None, split_ratio=0.8, seed=42):
    dataset = load_dataset("proxima-fusion/constellaration", "default")
    data = dataset["train"]
    
    # Filter invalid examples first
    data = data.filter(is_valid)
    
    # Optionally subset first N valid examples
    if subset_hyperparam is not None:
        data = data.select(range(min(subset_hyperparam, len(data))))
    
    # Shuffle indices
    indices = torch.arange(len(data))
    torch.manual_seed(seed)
    indices = indices[torch.randperm(len(indices))]
    
    # Split into train/test
    split = int(split_ratio * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_dataset = data.select(train_idx.tolist())
    test_dataset = data.select(test_idx.tolist())
    
    return train_dataset, test_dataset


class ConstellarationDataset(Dataset):
    """
    Custom Dataset that separates optimizable features (r_cos, z_sin) 
    from fixed features for post-training optimization.
    """
    def __init__(self, hf_dataset, separate_optimizable=False):
        """
        Args:
            hf_dataset: HuggingFace dataset
            separate_optimizable: If True, returns (optimizable_features, fixed_features, target)
                                 If False, returns (all_features, target) for training
        """
        self.data = []
        self.separate_optimizable = separate_optimizable
        
        # Pre-process and store as numpy arrays
        for sample in hf_dataset:
            try:
                # Extract optimizable features
                r_cos = np.array(sample["boundary.r_cos"], dtype=np.float32).flatten()
                z_sin = np.array(sample["boundary.z_sin"], dtype=np.float32).flatten()
                
                # Extract fixed features
                fixed_metrics = np.array([
                    float(sample["metrics.aspect_ratio"]),
                    float(sample["metrics.average_triangularity"]),
                    float(sample["metrics.edge_rotational_transform_over_n_field_periods"])
                ], dtype=np.float32)
                
                target = float(sample["metrics.max_elongation"])
                
                # Store separately for flexibility
                self.data.append({
                    'r_cos': r_cos,
                    'z_sin': z_sin,
                    'fixed': fixed_metrics,
                    'target': target
                })
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        if len(self.data) == 0:
            raise ValueError("No valid samples found in dataset")
        
        # Store feature dimensions for later use
        sample = self.data[0]
        self.r_cos_dim = len(sample['r_cos'])
        self.z_sin_dim = len(sample['z_sin'])
        self.fixed_dim = len(sample['fixed'])
        
        print(f"Successfully loaded {len(self.data)} samples")
        #print(f"Feature dimensions - r_cos: {self.r_cos_dim}, z_sin: {self.z_sin_dim}, fixed: {self.fixed_dim}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.separate_optimizable:
            # For optimization: return optimizable and fixed features separately
            optimizable = np.concatenate([sample['r_cos'], sample['z_sin']])
            return (
                torch.from_numpy(optimizable),
                torch.from_numpy(sample['fixed']),
                torch.tensor(sample['target'], dtype=torch.float32)
            )
        else:
            # For training: return all features concatenated
            features = np.concatenate([sample['r_cos'], sample['z_sin'], sample['fixed']])
            return (
                torch.from_numpy(features),
                torch.tensor(sample['target'], dtype=torch.float32)
            )


def collate_fn(batch):
    """Custom collate function that handles both training and optimization modes."""
    if len(batch[0]) == 3:
        # Optimization mode: (optimizable, fixed, target)
        optimizable, fixed, targets = zip(*batch)
        return (
            torch.stack(optimizable),
            torch.stack(fixed),
            torch.stack(targets).reshape(-1, 1)
        )
    else:
        # Training mode: (features, target)
        features, targets = zip(*batch)
        return torch.stack(features), torch.stack(targets).reshape(-1, 1)


# Example usage for training
def create_training_dataloader(train_dataset, batch_size=32):
    """Create dataloader for training with all features concatenated."""
    dataset = ConstellarationDataset(train_dataset, separate_optimizable=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# Example usage for optimization
def create_optimization_dataloader(test_dataset, batch_size=32):
    """Create dataloader for optimization with features separated."""
    dataset = ConstellarationDataset(test_dataset, separate_optimizable=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


