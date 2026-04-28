import torch
import torch.nn as nn


class DAM(nn.Module):
    """Dynamic Attention Mixer that predicts branch weights."""

    def __init__(self, channels: int, reduction: int = 16, num_branches: int = 3) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if reduction <= 0:
            raise ValueError("reduction must be > 0")
        if num_branches <= 1:
            raise ValueError("num_branches must be > 1")

        hidden = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, num_branches)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        pooled = self.gap(x).view(b, c)
        logits = self.fc2(self.relu(self.fc1(pooled)))
        weights = self.softmax(logits)
        return weights.view(b, -1, 1, 1)
