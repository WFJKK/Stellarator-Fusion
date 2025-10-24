import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import load_constellaration_dataset, ConstellarationDataset, collate_fn
from model import StellaratorNet
from config import device


def optimize_features(model_path, num_iterations=10, lr=0.01, batch_size=32):
    """
    Optimize r_cos and z_sin features to minimize the target (max_elongation)
    while keeping other features fixed.
    
    Args:
        model_path (str): Path to trained model weights
        num_iterations (int): Number of optimization iterations
        lr (float): Learning rate for feature optimization
        batch_size (int): Batch size
    
    Returns:
        optimized_features: Dictionary with optimized r_cos and z_sin for each sample
    """
    # Get data with separated features first
    _, test_data = load_constellaration_dataset()
    test_dataset = ConstellarationDataset(test_data, separate_optimizable=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Get dimensions from the dataset
    dims = {
        'r_cos_dim': test_dataset.r_cos_dim,
        'z_sin_dim': test_dataset.z_sin_dim,
        'fixed_dim': test_dataset.fixed_dim,
        'total_dim': test_dataset.r_cos_dim + test_dataset.z_sin_dim + test_dataset.fixed_dim
    }
    
    # Load trained model
    model = StellaratorNet(dims['total_dim']).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to eval mode - we're not training the model
    
    
    
    all_results = []
    
    # Process each batch
    for batch_idx, (optimizable, fixed, targets) in enumerate(test_loader):
        optimizable = optimizable.to(device)
        fixed = fixed.to(device)
        targets = targets.to(device)
        
        # Create optimizable parameters - detach from original and require grad
        optimizable_params = optimizable.clone().detach().requires_grad_(True)
        
        # Use Adam optimizer for the features
        optimizer = optim.Adam([optimizable_params], lr=lr)
        
        initial_loss = None
        
        # Optimize this batch
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Concatenate optimizable and fixed features
            combined_features = torch.cat([optimizable_params, fixed], dim=1)
            
            # Forward pass through model
            predictions = model(combined_features)
            
            # Loss: we want to minimize the prediction (max_elongation)
            loss = predictions.mean()
            
            if iteration == 0:
                initial_loss = loss.item()
            
            # Backward pass - only optimizable_params will get gradients
            loss.backward()
            optimizer.step()
            
            if (iteration + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}, Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.6f}")
        
        final_loss = loss.item()
        print(f"Batch {batch_idx+1} - Initial: {initial_loss:.6f}, Final: {final_loss:.6f}, "
              f"Improvement: {(initial_loss - final_loss):.6f}")
        
        # Store results
        r_cos_dim = dims['r_cos_dim']
        optimized_r_cos = optimizable_params[:, :r_cos_dim].detach().cpu()
        optimized_z_sin = optimizable_params[:, r_cos_dim:].detach().cpu()
        
        all_results.append({
            'r_cos': optimized_r_cos,
            'z_sin': optimized_z_sin,
            'fixed': fixed.cpu(),
            'original_target': targets.cpu(),
            'initial_prediction': initial_loss,
            'final_prediction': final_loss
        })
    
    return all_results