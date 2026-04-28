import torch
import torch.nn as nn

from .common import conv3x3
from .pa import PA


class _FRMStage(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )
        self.pa = PA(channels)
        self.conv = conv3x3(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.pa(x)
        return self.conv(x)


class FRM(nn.Module):
    """Feature Reconstruction Module.

    x2:  one stage (TransposedConv -> PA -> Conv)
    x4:  two staged upsampling.
    """

    def __init__(self, channels: int, scale: int = 2) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"FRM currently supports scale=2/4, got {scale}")

        num_stages = 1 if scale == 2 else 2
        self.stages = nn.Sequential(*[_FRMStage(channels) for _ in range(num_stages)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)
