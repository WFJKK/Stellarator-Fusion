import torch
from train import train_model, get_dataloaders
from model import StellaratorNet
from config import device,epochs 

# Get data loaders
train_loader, test_loader = get_dataloaders()

# Initialize model
input_dim = train_loader.dataset[0][0].shape[0]
model = StellaratorNet(input_dim).to(device)

# Train the model
trained_model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, num_epochs=epochs)





