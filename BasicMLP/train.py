import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import load_constellaration_dataset, ConstellarationDataset, collate_fn
from model import StellaratorNet
from config import device, learning_rate, batch_size, epochs, weight_decay


def train_model(model=None, train_loader=None, test_loader=None, num_epochs=epochs, lr=learning_rate, save_path="best_model.pth"):
    """
    Trains the model and saves the weights with the best test loss.
    Args:
        model (nn.Module): PyTorch model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test/validation data loader
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        save_path (str): Path to save the best model weights
    Returns:
        model: Trained PyTorch model
    """
    # If no model is provided, create one
    if model is None:
        train_data, _ = load_constellaration_dataset()
        # Use separate_optimizable=False for training
        train_dataset = ConstellarationDataset(train_data, separate_optimizable=False)
        input_dim = train_dataset[0][0].shape[0]
        model = StellaratorNet(input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_test_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # --- Evaluation ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * features.size(0)
        
        test_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        # Save model if test loss improved
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with test loss {best_test_loss:.4f} at epoch {epoch+1}")
    
    # Load best model weights before returning
    model.load_state_dict(torch.load(save_path))
    return model


def get_dataloaders(batch_size=batch_size, separate_optimizable=False):
    """
    Get dataloaders for training or optimization.
    
    Args:
        batch_size (int): Batch size
        separate_optimizable (bool): If False, returns concatenated features for training.
                                    If True, returns separated features for optimization.
    
    Returns:
        train_loader, test_loader: DataLoaders
    """
    train_data, test_data = load_constellaration_dataset(subset_hyperparam=None)
    
    train_dataset = ConstellarationDataset(train_data, separate_optimizable=separate_optimizable)
    test_dataset = ConstellarationDataset(test_data, separate_optimizable=separate_optimizable)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader


def get_feature_dimensions():
    """
    Helper function to get feature dimensions for optimization setup.
    
    Returns:
        dict: Dictionary with r_cos_dim, z_sin_dim, fixed_dim
    """
    # Load enough samples to ensure we get at least one valid sample
    train_data, _ = load_constellaration_dataset(subset_hyperparam=100)
    train_dataset = ConstellarationDataset(train_data, separate_optimizable=False)
    
    return {
        'r_cos_dim': train_dataset.r_cos_dim,
        'z_sin_dim': train_dataset.z_sin_dim,
        'fixed_dim': train_dataset.fixed_dim,
        'total_dim': train_dataset.r_cos_dim + train_dataset.z_sin_dim + train_dataset.fixed_dim
    }


def get_feature_dimensions_from_dataset(dataset):
    """
    Alternative: Get dimensions from an existing dataset instance.
    
    Args:
        dataset: ConstellarationDataset instance
    
    Returns:
        dict: Dictionary with r_cos_dim, z_sin_dim, fixed_dim
    """
    return {
        'r_cos_dim': dataset.r_cos_dim,
        'z_sin_dim': dataset.z_sin_dim,
        'fixed_dim': dataset.fixed_dim,
        'total_dim': dataset.r_cos_dim + dataset.z_sin_dim + dataset.fixed_dim
    }




