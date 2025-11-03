"""
constellaration_dataset.py

Utilities for loading, validating, and preparing the 
Proxima Fusion 'constellaration' dataset.

Includes:
    - Data validation and filtering
    - Train/test splitting
    - PyTorch Dataset and DataLoader utilities
"""

import math
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def is_valid(example: dict) -> bool:
    """
    Checks whether a dataset example contains all required metrics with valid values.

    Args:
        example (dict): A sample from the HuggingFace dataset.

    Returns:
        bool: True if all required keys exist and contain valid numerical or list data.
    """
    required_metrics = [
        "metrics.max_elongation",
        "metrics.average_triangularity",
        "metrics.edge_rotational_transform_over_n_field_periods",
        "boundary.n_field_periods",
        "metrics.aspect_ratio",
        "boundary.z_sin",
        "boundary.r_cos",
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
            if not all(isinstance(row, list) and len(row) > 0 for row in val):
                return False
        else:
            return False
    return True


def load_constellaration_dataset(
    subset_hyperparam: Optional[int] = None,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load and preprocess the Proxima Fusion 'constellaration' dataset.

    Args:
        subset_hyperparam (int, optional): Number of valid samples to select (for debugging).
        split_ratio (float): Fraction of data used for training.
        seed (int): Random seed for reproducible splits.

    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, test_dataset)
    """
    dataset = load_dataset("proxima-fusion/constellaration", "default")
    data = dataset["train"]

    # Filter invalid examples
    data = data.filter(is_valid)

    # Optional subset
    if subset_hyperparam is not None:
        data = data.select(range(min(subset_hyperparam, len(data))))

    # Shuffle and split
    indices = torch.arange(len(data))
    torch.manual_seed(seed)
    indices = indices[torch.randperm(len(indices))]

    split = int(split_ratio * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    train_dataset = data.select(train_idx.tolist())
    test_dataset = data.select(test_idx.tolist())

    return train_dataset, test_dataset


class ConstellarationDataset(Dataset):
    """
    Custom PyTorch Dataset for the Constellaration dataset.

    Can return either:
        - All features concatenated (for training)
        - Optimizable and fixed features separately (for optimization)
    """

    def __init__(self, hf_dataset, separate_optimizable: bool = False):
        """
        Args:
            hf_dataset: HuggingFace dataset containing samples.
            separate_optimizable (bool): 
                If True, returns (optimizable_features, fixed_features, target).
                If False, returns (all_features, target).
        """
        self.data = []
        self.separate_optimizable = separate_optimizable

        for sample in hf_dataset:
            try:
                r_cos = torch.tensor(sample["boundary.r_cos"], dtype=torch.float32).flatten()
                z_sin = torch.tensor(sample["boundary.z_sin"], dtype=torch.float32).flatten()

                fixed_metrics = torch.tensor(
                    [
                        float(sample["metrics.aspect_ratio"]),
                        float(sample["metrics.average_triangularity"]),
                        float(sample["metrics.edge_rotational_transform_over_n_field_periods"]),
                    ],
                    dtype=torch.float32,
                )

                target = torch.tensor(float(sample["metrics.max_elongation"]), dtype=torch.float32)

                self.data.append(
                    {"r_cos": r_cos, "z_sin": z_sin, "fixed": fixed_metrics, "target": target}
                )

            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        if not self.data:
            raise ValueError("No valid samples found in dataset")

        sample = self.data[0]
        self.r_cos_dim = sample["r_cos"].numel()
        self.z_sin_dim = sample["z_sin"].numel()
        self.fixed_dim = sample["fixed"].numel()

        print(f"Successfully loaded {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        if self.separate_optimizable:
            optimizable = torch.cat([sample["r_cos"], sample["z_sin"]])
            return optimizable, sample["fixed"], sample["target"]

        features = torch.cat([sample["r_cos"], sample["z_sin"], sample["fixed"]])
        return features, sample["target"]


def collate_fn(batch):
    """
    Custom collate function that supports both training and optimization modes.

    Args:
        batch (list): A batch of dataset samples.

    Returns:
        tuple: Tensors formatted for model input.
    """
    if len(batch[0]) == 3:
        optimizable, fixed, targets = zip(*batch)
        return torch.stack(optimizable), torch.stack(fixed), torch.stack(targets).reshape(-1, 1)

    features, targets = zip(*batch)
    return torch.stack(features), torch.stack(targets).reshape(-1, 1)


def create_training_dataloader(train_dataset, batch_size: int = 32) -> DataLoader:
    """
    Create a DataLoader for model training.

    Args:
        train_dataset: HuggingFace dataset for training.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: Batched training data with all features concatenated.
    """
    dataset = ConstellarationDataset(train_dataset, separate_optimizable=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def create_optimization_dataloader(test_dataset, batch_size: int = 32) -> DataLoader:
    """
    Create a DataLoader for optimization tasks.

    Args:
        test_dataset: HuggingFace dataset for evaluation/optimization.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: Batched data with optimizable and fixed features separated.
    """
    dataset = ConstellarationDataset(test_dataset, separate_optimizable=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



