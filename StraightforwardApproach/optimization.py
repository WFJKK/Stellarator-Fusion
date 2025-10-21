import torch
from helper import StellaratorNet
import config

def optimize_fourier_coefficients(
    model_path,
    fixed_len,
    fixed_features,
    num_iterations=500,
    lr=0.01,
    clip_min=-1.0,
    clip_max=1.0,
    dropout_prob=0.2,
):
    device = config.device

    input_size = fixed_len*4 + len(fixed_features)
    model = StellaratorNet(input_size, dropout_prob=dropout_prob).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = torch.randn(fixed_len*4, device=device, requires_grad=True)
    fixed_features = torch.tensor(fixed_features, device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam([x], lr=lr)
    best_x = None
    best_score = float("inf")

    for i in range(num_iterations):
        optimizer.zero_grad()
        input_vec = torch.cat([x, fixed_features])
        pred = model(input_vec.unsqueeze(0))
        loss = pred
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(clip_min, clip_max)
        if pred.item() < best_score:
            best_score = pred.item()
            best_x = x.detach().clone()
    return best_x, best_score
