import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def is_valid(sample: dict) -> bool:
    """
    Check if a dataset sample is valid by verifying required keys and values.

    A sample is considered invalid if any required key is missing, 
    contains NaN/Inf values (for floats), or empty lists.

    Args:
        sample (dict): A single dataset sample.

    Returns:
        bool: True if the sample is valid, False otherwise.
    """
    required_keys = [
        "metrics.max_elongation",
        "metrics.average_triangularity",
        "metrics.edge_rotational_transform_over_n_field_periods",
        "metrics.aspect_ratio",
        "boundary.n_field_periods",
        "boundary.r_cos",
        "boundary.z_sin"
    ]

    for key in required_keys:
        val = sample.get(key)
        if val is None:
            return False
        if isinstance(val, float) and (torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val))):
            return False
        if isinstance(val, list) and len(val) == 0:
            return False

    return True


def load_constellaration_dataset(subset_hyperparam: int = None, split_ratio: float = 0.8, seed: int = 42):
    """
    Load, filter, optionally subset, and split the Constelleration dataset.

    Args:
        subset_hyperparam (int, optional): Limit dataset size to this number of samples.
        split_ratio (float, optional): Fraction of data to use for training.
        seed (int, optional): Random seed for reproducible splits.

    Returns:
        tuple: (train_dataset, test_dataset) as HuggingFace datasets.
    """
    dataset = load_dataset("proxima-fusion/constellaration", "default")["train"]
    dataset = dataset.filter(is_valid, batched=False, num_proc=1)

    if subset_hyperparam is not None:
        dataset = dataset.select(range(min(subset_hyperparam, len(dataset))))

    indices = torch.arange(len(dataset))
    torch.manual_seed(seed)
    indices = indices[torch.randperm(len(indices))]

    split = int(split_ratio * len(indices))
    train_dataset = dataset.select(indices[:split].tolist())
    test_dataset = dataset.select(indices[split:].tolist())

    return train_dataset, test_dataset


def create_boundary(sample: dict, n_theta: int = 15, n_phi: int = 15, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct the 2D boundary surface (R and Z) from Fourier-like coefficients.

    Args:
        sample (dict): Dataset sample containing 'boundary' fields.
        n_theta (int, optional): Number of discretization points along theta.
        n_phi (int, optional): Number of discretization points along phi.
        device (torch.device, optional): Device to place tensors on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Reconstructed (R, Z) boundary tensors.
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
    Extract the target variable (max_elongation) from a dataset sample.

    Args:
        sample (dict): Dataset sample containing 'metrics.max_elongation'.
        device (torch.device, optional): Device to place tensor on.

    Returns:
        torch.Tensor | None: Target tensor, or None if missing.
    """
    target = sample.get("metrics.max_elongation")
    if target is None:
        return None

    y = torch.tensor(target, dtype=torch.float32)
    if device:
        y = y.to(device)
    return y


def extract_mlpfeatures(sample: dict, device: torch.device = None) -> torch.Tensor | None:
    """
    Extract selected MLP input features from a dataset sample.

    Args:
        sample (dict): Dataset sample containing metrics.
        device (torch.device, optional): Device to place tensor on.

    Returns:
        torch.Tensor | None: Feature tensor, or None if any feature is missing.
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


class ConstellerationCNNDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Constelleration dataset.

    Returns:
        tuple: (boundary_tensor, mlp_features, target) for each sample.
    """
    def __init__(self, hf_dataset, device: torch.device = None):
        self.data = hf_dataset
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        boundary = create_boundary(sample, device=self.device)
        mlp_features = extract_mlpfeatures(sample, device=self.device)
        target = extract_target(sample, device=self.device)
        return boundary, mlp_features, target

