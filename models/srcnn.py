import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """SRCNN for image super-resolution.

    Input must already be bicubic-upsampled to target size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features_1: int = 64,
        num_features_2: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_features_1,
            kernel_size=9,
            stride=1,
            padding=4,
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=num_features_1,
            out_channels=num_features_2,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels=num_features_2,
            out_channels=out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"SRCNN expects 4D input [B, C, H, W], but got shape={tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"SRCNN expects input channels={self.in_channels}, but got {x.shape[1]}"
            )

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x
