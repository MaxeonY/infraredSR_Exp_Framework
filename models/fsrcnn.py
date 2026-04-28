import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    """FSRCNN for single-channel infrared super-resolution."""

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        d: int = 56,
        s: int = 12,
        m: int = 4,
    ) -> None:
        super().__init__()

        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be > 0")
        if d <= 0 or s <= 0 or m <= 0:
            raise ValueError("d, s, m must all be > 0")

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(num_parameters=d),
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1, stride=1, padding=0),
            nn.PReLU(num_parameters=s),
        )

        mapping = []
        for _ in range(m):
            mapping.extend(
                [
                    nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1),
                    nn.PReLU(num_parameters=s),
                ]
            )
        self.mapping = nn.Sequential(*mapping)

        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1, stride=1, padding=0),
            nn.PReLU(num_parameters=d),
        )

        self.deconv = nn.ConvTranspose2d(
            in_channels=d,
            out_channels=out_channels,
            kernel_size=9,
            stride=scale,
            padding=4,
            output_padding=scale - 1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FSRCNN expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"FSRCNN expects {self.in_channels} channels, got {x.shape[1]}"
            )

        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x
