"""
config.py

Global hyperparameters and device configuration for training models.
"""


import torch

# --- Hyperparameters ---
subset_hyperparam = None #None means all samples
learning_rate = 1e-3
batch_size = 32
epochs = 30
dropout_prob = 0.2
weight_decay = 1e-4

# --- Device selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


