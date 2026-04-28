import torch
import torch.nn as nn

from .common import conv1x1, conv3x3


class PA(nn.Module):
    """Pixel Attention branch with residual refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")

        self.pixel_gate = nn.Sequential(
            conv1x1(channels, channels),
            nn.Sigmoid(),
        )
        self.feature_conv = conv3x3(channels, channels)
        self.refine_conv = conv3x3(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.pixel_gate(x)
        feat = self.feature_conv(x)
        mixed = feat * gate
        mixed = mixed + x
        return self.refine_conv(mixed)
