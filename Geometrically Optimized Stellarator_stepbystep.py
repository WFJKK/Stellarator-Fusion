from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from evaluation_scripts import compute_max_elongation

# --- Hyperparameters ---
fixed_len = 15          # how many Fourier coefficients to take
subset_hyperparam = 10 #how many samples to use. None means all !
learning_rate = 1e-3
batch_size = 32
num_epochs = 1000

# --- 0.) Device selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# --- 1.) Import dataset ---
dataset = load_dataset("proxima-fusion/constellaration", "default")
train_data = dataset['train']

# Shuffle indices and split 80% train / 20% test
indices = np.arange(len(train_data))
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, test_idx = indices[:split], indices[split:]
train_dataset = train_data.select(train_idx)
test_dataset = train_data.select(test_idx)

# If subset_hyperparam is set, reduce the train dataset
if subset_hyperparam is not None:
    train_dataset = train_dataset.select(range(min(subset_hyperparam, len(train_dataset))))

# --- 2.) Fix input features ---
def fix_length(arr, length):
    arr = np.array(arr, dtype=float).flatten()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) < length:
        arr = np.pad(arr, (0, length - len(arr)), mode='constant')
    else:
        arr = arr[:length]
    return arr

def extract_features(sample, length=fixed_len):
    r_cos = fix_length(sample['boundary.r_cos'], length)
    r_sin = fix_length(sample['boundary.r_sin'], length)
    z_cos = fix_length(sample['boundary.z_cos'], length)
    z_sin = fix_length(sample['boundary.z_sin'], length)
    return np.concatenate([r_cos, r_sin, z_cos, z_sin])

def prepare_dataset(dataset):
    X_list, y_list = [], []
    for s in dataset:
        val = s['metrics.max_elongation']
        if val is None:
            continue
        try:
            y_list.append(float(val))
            X_list.append(extract_features(s))
        except:
            continue
    X = torch.tensor(np.array(X_list, dtype=np.float32), dtype=torch.float32, device=device)
    y = torch.tensor(np.array(y_list, dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=device)
    return TensorDataset(X, y)

train_tensor_dataset = prepare_dataset(train_dataset)
test_tensor_dataset = prepare_dataset(test_dataset)

train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

print("Train samples:", len(train_tensor_dataset))
print("Test samples:", len(test_tensor_dataset))

# --- 3.) Neural Network ---


class StellaratorNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Layer 1
        self.fc1 = nn.Linear(input_size, 64)
        self.act1 = nn.ReLU()
        
        # Layer 2
        self.fc2 = nn.Linear(64, 64)  # match dimensions for skip
        self.act2 = nn.ReLU()
        
        # Layer 3
        self.fc3 = nn.Linear(64, 32)
        self.act3 = nn.ReLU()
        
        # Output
        self.fc4 = nn.Linear(32, 1)  # linear output

        # Optional: projection for skip connection from input to hidden
        self.skip1 = nn.Linear(input_size, 64) if input_size != 64 else nn.Identity()

    def forward(self, x):
        # First layer + skip from input
        x1 = self.act1(self.fc1(x) + self.skip1(x))  # residual connection from input
        
        # Second layer with residual
        x2 = self.act2(self.fc2(x1) + x1)  # residual
        
        # Third layer
        x3 = self.act3(self.fc3(x2))
        
        # Output layer
        out = self.fc4(x3)  # raw output
        return out


input_size = train_tensor_dataset[0][0].shape[0]
model = StellaratorNet(input_size).to(device)

# --- 4.) Training ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    
    # --- Testing ---
    model.eval()
    test_loss_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            test_loss_total += loss.item() * xb.size(0)
    test_loss = test_loss_total / len(test_loader.dataset)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")


