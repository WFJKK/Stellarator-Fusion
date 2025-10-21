from optimization import optimize_fourier_coefficients
from utils_output import format_fourier_as_json
import config

if __name__ == "__main__":
    # example fixed metrics
    fixed_metrics = (1.25, 0.45, 0.82)

    # optimize Fourier coefficients
    best_x, best_score = optimize_fourier_coefficients(
        model_path="best_model.pth",
        fixed_len=config.fixed_len,
        fixed_features=fixed_metrics,
        num_iterations=500,
        lr=0.01,
        clip_min=-1.0,
        clip_max=1.0,
        dropout_prob=config.dropout_prob,
    )

    # format Fourier coefficients as JSON string
    boundary_json_str = format_fourier_as_json(best_x, fixed_len=config.fixed_len)

    # print results
    print("Best predicted elongation:", best_score)
    print("Boundary JSON:", boundary_json_str)

