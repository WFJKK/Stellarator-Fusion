import json
import torch

def save_r_cos_z_sin_to_json(all_results, save_path="optimized_features.json"):
    """
    Save r_cos and z_sin from optimization results to a JSON file in the current folder.
    Converts tensors to lists automatically.
    """
    json_data = []

    for batch in all_results:
        r_cos = batch['r_cos']
        z_sin = batch['z_sin']

        # Convert tensors to lists if needed
        if isinstance(r_cos, torch.Tensor):
            r_cos = r_cos.detach().cpu().tolist()
        if isinstance(z_sin, torch.Tensor):
            z_sin = z_sin.detach().cpu().tolist()

        json_data.append({
            'r_cos': r_cos,
            'z_sin': z_sin
        })

    # Save directly to file
    with open(save_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"r_cos and z_sin saved to {save_path}")


