import numpy as np
import torch
from torch.utils.data import TensorDataset
from datasets import load_dataset


def load_constellaration_dataset(subset_hyperparam=None, split_ratio=0.8, seed=42):
    dataset = load_dataset("proxima-fusion/constellaration", "default")
    train_data = dataset["train"]

    indices = np.arange(len(train_data))
    np.random.seed(seed)
    np.random.shuffle(indices)

    split = int(split_ratio * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    train_dataset = train_data.select(train_idx)
    test_dataset = train_data.select(test_idx)

    if subset_hyperparam is not None:
        train_dataset = train_dataset.select(range(min(subset_hyperparam, len(train_dataset))))

    return train_dataset, test_dataset


def fix_length(arr, length):
    arr = np.array(arr, dtype=float).flatten()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) < length:
        arr = np.pad(arr, (0, length - len(arr)), mode="constant")
    else:
        arr = arr[:length]
    return arr


def extract_features(sample, length):
    # Fourier boundary coefficients
    r_cos = fix_length(sample["boundary.r_cos"], length)
    r_sin = fix_length(sample["boundary.r_sin"], length)
    z_cos = fix_length(sample["boundary.z_cos"], length)
    z_sin = fix_length(sample["boundary.z_sin"], length)

    # Add scalar metrics as extra features
    metrics = [
        sample.get("metrics.aspect_ratio", 0.0),
        sample.get("metrics.average_triangularity", 0.0),
        sample.get("metrics.edge_rotational_transform", 0.0),
    ]
    metrics = np.nan_to_num(np.array(metrics, dtype=float), nan=0.0)

    # Concatenate all features
    return np.concatenate([r_cos, r_sin, z_cos, z_sin, metrics])


def prepare_tensor_dataset(dataset, device, fixed_len):
    X_list, y_list = [], []
    for s in dataset:
        val = s["metrics.max_elongation"]
        if val is None:
            continue
        try:
            X_list.append(extract_features(s, fixed_len))
            y_list.append(float(val))
        except Exception:
            continue

    X = torch.tensor(np.array(X_list, dtype=np.float32), device=device)
    y = torch.tensor(np.array(y_list, dtype=np.float32).reshape(-1, 1), device=device)
    return TensorDataset(X, y)

