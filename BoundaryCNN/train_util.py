import torch
from torch import nn, optim
from torch.utils.data import DataLoader

def train_model(
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    weight_decay: float,
    batch_size: int = 32,
    num_epochs: int = 3,
    lr: float = 1e-3,
    device: torch.device = None,
    save_path: str = "best_model.pth",
) -> nn.Module:
    """
    Train a PyTorch model using the provided training and testing datasets.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_dataset (Dataset): Dataset for training.
        test_dataset (Dataset): Dataset for evaluation.
        weight_decay (float): Weight decay for the AdamW optimizer.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_epochs (int, optional): Number of training epochs. Defaults to 3.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        device (torch.device, optional): Device to run training on. Defaults to GPU if available.
        save_path (str, optional): Path to save the best model checkpoint. Defaults to "best_model.pth".

    Returns:
        nn.Module: The trained model (best state saved during training).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_test_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for boundary_tuple, mlp_features, target in train_loader:
            R, Z = boundary_tuple
            if isinstance(R, list) or R.dim() == 2:
                R = R.unsqueeze(0)
                Z = Z.unsqueeze(0)
            boundary_tuple_batch = (R.to(device), Z.to(device))
            mlp_features = mlp_features.to(device)
            target = target.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(boundary_tuple_batch, mlp_features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * target.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for boundary_tuple, mlp_features, target in test_loader:
                R, Z = boundary_tuple
                if isinstance(R, list) or R.dim() == 2:
                    R = R.unsqueeze(0)
                    Z = Z.unsqueeze(0)
                boundary_tuple_batch = (R.to(device), Z.to(device))
                mlp_features = mlp_features.to(device)
                target = target.to(device).unsqueeze(1)
                
                outputs = model(boundary_tuple_batch, mlp_features)
                loss = criterion(outputs, target)
                test_loss += loss.item() * target.size(0)
        
        test_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
    
    return model


