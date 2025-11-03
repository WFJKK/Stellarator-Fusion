"""
This script performs feature optimization for a trained StellaratorNet model.

It optimizes the 'r_cos' and 'z_sin' features of the dataset to minimize
the target (e.g., max_elongation) while keeping other features fixed.

The optimized features are then saved both as a PyTorch tensor file (.pt)
and as a JSON file for easy inspection or downstream use.
"""



from optimization import optimize_features
from utils_output import save_r_cos_z_sin_to_json
import torch
from config_optimization import lr_opti,iterations


if __name__ == "__main__":
    
    results = optimize_features(
        model_path="best_model.pth",
        num_iterations=iterations,
        lr=lr_opti,
        batch_size=150
    )

    
    torch.save(results, "optimized_features.pt")
    print("Saved optimized features to optimized_features.pt")

    
    save_r_cos_z_sin_to_json(results, "optimized_features.json")

