import torch

config = {
    "batch_size": 4,
    "n_theta": 16,
    "n_phi": 16,
    "mlp_input_dim": 3,
    "hidden_dim": 64,
    "num_outputs": 1,
    "learning_rate": 1e-3,
    "num_epochs": 100,
    "weight_decay": 1e-3
}

# --- Device selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)