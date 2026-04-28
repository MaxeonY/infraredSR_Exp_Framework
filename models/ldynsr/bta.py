import torch
import torch.nn as nn

from .common import conv1x1, conv3x3


class BTA(nn.Module):
    """Brightness-Texture Attention branch.

    Paper descriptions on tensor-level fusion are not fully explicit.
    This implementation is a reasonable reproduction completion.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if reduction <= 0:
            raise ValueError("reduction must be > 0")

        hidden = max(1, channels // reduction)
        top_channels = channels // 2
        bottom_channels = channels - top_channels
        if top_channels <= 0 or bottom_channels <= 0:
            raise ValueError(
                f"BTA requires channels >= 2 for split-concat design, got {channels}"
            )

        self.top_channels = top_channels
        self.bottom_channels = bottom_channels

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.brightness_branch = nn.Sequential(
            conv1x1(channels, hidden),
            nn.ReLU(inplace=True),
            conv1x1(hidden, top_channels),
            nn.Sigmoid(),
        )

        # Figure (c): 3x3 conv -> 1x1 conv -> sigmoid.
        self.texture_branch = nn.Sequential(
            conv3x3(channels, bottom_channels),
            conv1x1(bottom_channels, bottom_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        brightness = self.brightness_branch(self.gap(x))
        texture = self.texture_branch(x)
        brightness_map = brightness.expand(-1, -1, x.size(2), x.size(3))
        attn = torch.cat([brightness_map, texture], dim=1)
        return x * attn
