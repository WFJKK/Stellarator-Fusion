import json
import numpy as np

def format_fourier_as_json(fourier_coeffs, fixed_len=15):
    """
    Converts a flat Fourier coefficient vector into a JSON-style dictionary
    with keys: r_cos, r_sin, z_cos, z_sin. Each is a list of lists
    corresponding to the number of modes (4 here) and fixed_len coefficients.
    """

    fourier_coeffs = fourier_coeffs.cpu().numpy() if hasattr(fourier_coeffs, 'cpu') else np.array(fourier_coeffs)

    # split into four sets
    r_cos = fourier_coeffs[0*fixed_len:1*fixed_len].tolist()
    r_sin = fourier_coeffs[1*fixed_len:2*fixed_len].tolist()
    z_cos = fourier_coeffs[2*fixed_len:3*fixed_len].tolist()
    z_sin = fourier_coeffs[3*fixed_len:4*fixed_len].tolist()

    # wrap in nested lists as in your example
    # here each coefficient list is wrapped in an outer list
    # (can be adapted if multiple modes needed)
    boundary_json = {
        "r_cos": [r_cos],
        "r_sin": [r_sin],
        "z_cos": [z_cos],
        "z_sin": [z_sin]
    }

    # convert to stringified JSON
    return json.dumps(boundary_json)
