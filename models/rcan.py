import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, n_feats: int, reduction: int = 16) -> None:
        super().__init__()
        if n_feats <= 0:
            raise ValueError("n_feats must be > 0")
        if reduction <= 0:
            raise ValueError("reduction must be > 0")

        hidden = max(1, n_feats // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(n_feats, hidden, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_feats, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.mlp(self.avg_pool(x))
        return x * weight


class RCAB(nn.Module):
    def __init__(
        self,
        n_feats: int = 64,
        reduction: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            ChannelAttention(n_feats=n_feats, reduction=reduction),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ResidualGroup(nn.Module):
    def __init__(
        self,
        n_resblocks: int,
        n_feats: int = 64,
        reduction: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if n_resblocks <= 0:
            raise ValueError("n_resblocks must be > 0")

        layers = [
            RCAB(n_feats=n_feats, reduction=reduction, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        layers.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class UpsampleBlock(nn.Module):
    def __init__(self, scale: int, n_feats: int) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"RCAN currently supports scale=2/4, got {scale}")

        layers = []
        if scale == 2:
            layers.extend(
                [
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                ]
            )
        else:
            for _ in range(2):
                layers.extend(
                    [
                        nn.Conv2d(
                            n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1
                        ),
                        nn.PixelShuffle(2),
                    ]
                )
        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class RCAN(nn.Module):
    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        n_resgroups: int = 5,
        n_resblocks: int = 10,
        n_feats: int = 64,
        reduction: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"RCAN currently supports scale=2/4, got {scale}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be > 0")
        if n_resgroups <= 0 or n_resblocks <= 0 or n_feats <= 0:
            raise ValueError("n_resgroups, n_resblocks, n_feats must be > 0")

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1)
        groups = [
            ResidualGroup(
                n_resblocks=n_resblocks,
                n_feats=n_feats,
                reduction=reduction,
                res_scale=res_scale,
            )
            for _ in range(n_resgroups)
        ]
        groups.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*groups)
        self.tail = nn.Sequential(
            UpsampleBlock(scale=scale, n_feats=n_feats),
            nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"RCAN expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"RCAN expects {self.in_channels} channels, got {x.shape[1]}"
            )

        feat = self.head(x)
        res = self.body(feat) + feat
        return self.tail(res)
