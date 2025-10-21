import torch.nn as nn

class StellaratorNet(nn.Module):
    def __init__(self, input_size, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.skip1 = nn.Linear(input_size, 64) if input_size != 64 else nn.Identity()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x1 = self.act(self.fc1(x) + self.skip1(x))
        x1 = self.dropout1(x1)
        x2 = self.act(self.fc2(x1) + x1)
        x2 = self.dropout2(x2)
        x3 = self.act(self.fc3(x2))
        x3 = self.dropout3(x3)
        out = self.fc4(x3)
        return out
