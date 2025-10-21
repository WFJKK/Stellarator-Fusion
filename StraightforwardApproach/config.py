import torch

# --- Hyperparameters ---
fixed_len = 15
subset_hyperparam = None #None means all samples
learning_rate = 1e-3
batch_size = 32
num_epochs = 1
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