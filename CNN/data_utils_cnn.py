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
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

def is_valid(sample: dict) -> bool:
    """
    Check if a sample has valid, non-empty nested list structures.
    """
    for val in sample.values():
        if isinstance(val, list) and all(isinstance(row, list) and len(row) > 0 for row in val):
            continue
        else:
            return False
    return True

def load_constellaration_dataset(subset_hyperparam: int = None, split_ratio: float = 0.8, seed: int = 42):
    """
    Load the Constelleration dataset, filter invalid examples, optionally subset,
    and split into training and testing sets.
    
    Returns:
        tuple: (train_dataset, test_dataset) as HuggingFace datasets
    """
    dataset = load_dataset("proxima-fusion/constellaration", "default")
    data = dataset["train"]

    data = data.filter(is_valid)

    if subset_hyperparam is not None:
        data = data.select(range(min(subset_hyperparam, len(data))))

    indices = torch.arange(len(data))
    torch.manual_seed(seed)
    indices = indices[torch.randperm(len(indices))]

    split = int(split_ratio * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    train_dataset = data.select(train_idx.tolist())
    test_dataset = data.select(test_idx.tolist())

    return train_dataset, test_dataset

def extract_mlpfeatures(sample: dict, device: torch.device = None) -> torch.Tensor | None:
    """
    Extracts selected MLP features from a dataset sample.
    """
    features = [
        sample.get("metrics.aspect_ratio"),
        sample.get("metrics.average_triangularity"),
        sample.get("metrics.edge_rotational_transform_over_n_field_periods"),
    ]
    if any(f is None for f in features):
        return None

    x = torch.tensor(features, dtype=torch.float32)
    if device:
        x = x.to(device)
    return x

def create_boundary(sample: dict, n_theta: int = 15, n_phi: int = 15, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstructs the boundary surface functions R(theta, phi) and Z(theta, phi)
    from Fourier-like mode coefficients stored in the sample.
    """
    n_fp = torch.tensor(sample['boundary.n_field_periods'], dtype=torch.float32)
    r_cos = torch.as_tensor(sample['boundary.r_cos'], dtype=torch.float32)
    z_sin = torch.as_tensor(sample['boundary.z_sin'], dtype=torch.float32)
    assert r_cos.shape == z_sin.shape, "Coefficient shapes must match"
    M, N = r_cos.shape

    theta = torch.linspace(0, 2 * torch.pi, n_theta)
    phi = torch.linspace(0, 2 * torch.pi, n_phi)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    theta = theta.unsqueeze(0).unsqueeze(0)
    phi = phi.unsqueeze(0).unsqueeze(0)

    m = torch.arange(M, dtype=torch.float32).view(M, 1, 1, 1)
    n = torch.arange(N, dtype=torch.float32).view(1, N, 1, 1)

    R = (r_cos.view(M, N, 1, 1) * torch.cos(m * theta - n * n_fp * phi)).sum(dim=(0, 1))
    Z = (z_sin.view(M, N, 1, 1) * torch.sin(m * theta - n * n_fp * phi)).sum(dim=(0, 1))

    if device:
        R = R.to(device)
        Z = Z.to(device)

    return R, Z

def extract_target(sample: dict, device: torch.device = None) -> torch.Tensor | None:
    """
    Extracts the target variable (max_elongation) from a sample.
    """
    target = sample.get("metrics.max_elongation")
    if target is None:
        return None
    y = torch.tensor(target, dtype=torch.float32)
    if device:
        y = y.to(device)
    return y

class ConstellerationCNNDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Constelleration dataset.
    Returns tuples of (boundary_tensor, mlp_features, target).
    """
    def __init__(self, hf_dataset, device: torch.device = None):
        self.data = hf_dataset
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        boundary = create_boundary(sample)
        mlp_features = extract_mlpfeatures(sample, device=self.device)
        target = extract_target(sample, device=self.device)
        return boundary, mlp_features, target






