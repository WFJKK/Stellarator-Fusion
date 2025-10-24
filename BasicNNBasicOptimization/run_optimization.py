from optimization import optimize_features
from utils_output import save_r_cos_z_sin_to_json
import torch
from config_optimization import lr_opti,iterations

if __name__ == "__main__":
    # Run feature optimization
    results = optimize_features(
        model_path="best_model.pth",
        num_iterations=iterations,
        lr=lr_opti,
        batch_size=150
    )

    # Save as PyTorch file
    torch.save(results, "optimized_features.pt")
    print("Saved optimized features to optimized_features.pt")

    # Save r_cos and z_sin as JSON
    save_r_cos_z_sin_to_json(results, "optimized_features.json")

