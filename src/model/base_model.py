import torch.nn as nn


class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2)
        )
        self.res1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
        )

        self.res2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
        )
        self.out = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = self.input(x)
        x = self.res1(x) + x
        x = self.res2(x) + x
        x = self.out(x)
        return x
