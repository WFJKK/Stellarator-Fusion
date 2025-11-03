import torch
import torch.nn as nn
import torch.nn.functional as F

class PeriodicConv2d(nn.Module):
    """
    2D convolution with circular (periodic) padding in both dimensions.
    Preserves periodicity along θ and φ coordinates.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        return self.conv(x)

class BoundaryCNN(nn.Module):
    """
    CNN for processing 2D boundary maps with multi-scale pooling.
    Takes a tuple of (R, Z) boundary tensors as input.
    """
    def __init__(self, n_theta: int, n_phi: int, num_outputs: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv1 = PeriodicConv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = PeriodicConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = PeriodicConv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = PeriodicConv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256 * 2, num_outputs)

    def forward(self, boundary_tuple: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        R, Z = boundary_tuple
        if R.dim() == 2:
            R = R.unsqueeze(0)
            Z = Z.unsqueeze(0)

        x = torch.stack([R, Z], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x_avg = self.gap(x).view(x.size(0), -1)
        x_max = self.gmp(x).view(x.size(0), -1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.dropout(x)
        return self.fc(x)

class MLP(nn.Module):
    """
    Multi-layer perceptron with batch normalization and dropout.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)

        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

class CombinedModel(nn.Module):
    """
    Combines a BoundaryCNN and an MLP for joint feature processing.
    Concatenates CNN and MLP outputs before final fully connected layers.
    """
    def __init__(self, n_theta: int, n_phi: int, mlp_input_dim: int = 3, hidden_dim: int = 128, num_outputs: int = 1):
        super().__init__()
        self.boundary_cnn = BoundaryCNN(n_theta, n_phi, hidden_dim)
        self.mlp = MLP(mlp_input_dim, hidden_dim, hidden_dim)

        self.final_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_outputs)
        )

    def forward(self, boundary_tuple: tuple[torch.Tensor, torch.Tensor], mlp_features: torch.Tensor) -> torch.Tensor:
        cnn_out = self.boundary_cnn(boundary_tuple)
        mlp_out = self.mlp(mlp_features)
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        return F.softplus(self.final_fc(combined))

          
