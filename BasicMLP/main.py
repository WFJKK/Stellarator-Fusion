"""
main.py

Entry point for training the StellaratorNet model on the Constellaration dataset.
Loads dataloaders, initializes the model, and runs the training loop.
"""


from train import train_model, get_dataloaders
from model import StellaratorNet
from config import device,epochs 


train_loader, test_loader = get_dataloaders()


input_dim = train_loader.dataset[0][0].shape[0]
model = StellaratorNet(input_dim).to(device)

trained_model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, num_epochs=epochs)





