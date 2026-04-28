"""EDSR with Adaptive Receptive Field (ARF) kernel sizes.

This script is intentionally separated from ``edsr.py`` so both variants can
coexist and be compared side by side.

ARF follows Section 2 ("basic explicit formula") in
``Adaptive Reception Field.md``:

    d_i = (i - 1) / (L - 1)
    c_i = (log C_i - log C_min) / (log C_max - log C_min)
    s_i = alpha * d_i + beta * c_i + gamma * d_i * c_i
    t_i = t_min + (t_max - t_min) * sigmoid(s_i)
    k_i = Q_odd(t_i)

where ``Q_odd`` quantizes to the nearest odd kernel size.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _nearest_odd(x: float, k_min: int = 3, k_max: int = 11) -> int:
    """Quantize a real value to the nearest odd integer in [k_min, k_max]."""
    k = int(round(x))
    if k % 2 == 0:
        k = k + 1 if x >= k else k - 1

    k = max(k_min, min(k_max, k))
    if k % 2 == 0:
        k = k + 1 if k < k_max else k - 1
    return k


def compute_adaptive_kernel_sizes(
    channels: Sequence[int],
    alpha: float = 2.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    t_min: float = 3.0,
    t_max: float = 11.0,
) -> List[int]:
    """Compute per-layer odd kernel sizes from ARF formula."""
    L = len(channels)
    if L == 0:
        return []

    if L == 1:
        d_list = [0.5]
    else:
        d_list = [i / (L - 1) for i in range(L)]

    log_c = [math.log(max(int(c), 1)) for c in channels]
    log_min = min(log_c)
    log_max = max(log_c)
    denom = log_max - log_min
    if denom < 1e-8:
        c_list = [0.5] * L
    else:
        c_list = [(lc - log_min) / denom for lc in log_c]

    odd_min = _nearest_odd(min(t_min, t_max), k_min=1, k_max=99)
    odd_max = _nearest_odd(max(t_min, t_max), k_min=1, k_max=99)

    kernels: List[int] = []
    for d_i, c_i in zip(d_list, c_list):
        s_i = alpha * d_i + beta * c_i + gamma * d_i * c_i
        sigma_s = 1.0 / (1.0 + math.exp(-s_i))
        t_i = t_min + (t_max - t_min) * sigma_s
        kernels.append(_nearest_odd(t_i, k_min=odd_min, k_max=odd_max))
    return kernels


class AdaptiveResidualBlock(nn.Module):
    def __init__(
        self,
        n_feats: int,
        kernel_size_1: int,
        kernel_size_2: int,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feats,
                n_feats,
                kernel_size=kernel_size_1,
                stride=1,
                padding=kernel_size_1 // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                n_feats,
                n_feats,
                kernel_size=kernel_size_2,
                stride=1,
                padding=kernel_size_2 // 2,
            ),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class AdaptiveUpsampleBlock(nn.Module):
    def __init__(self, scale: int, n_feats: int, kernel_sizes: Sequence[int]) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"EDSR_ARF currently supports scale=2/4, got {scale}")

        expected = 1 if scale == 2 else 2
        if len(kernel_sizes) != expected:
            raise ValueError(
                f"Upsample block needs {expected} kernel size(s) at scale={scale}, "
                f"got {len(kernel_sizes)}"
            )

        layers = []
        if scale == 2:
            k = int(kernel_sizes[0])
            layers.extend(
                [
                    nn.Conv2d(
                        n_feats,
                        n_feats * 4,
                        kernel_size=k,
                        stride=1,
                        padding=k // 2,
                    ),
                    nn.PixelShuffle(2),
                ]
            )
        else:
            for k in kernel_sizes:
                k = int(k)
                layers.extend(
                    [
                        nn.Conv2d(
                            n_feats,
                            n_feats * 4,
                            kernel_size=k,
                            stride=1,
                            padding=k // 2,
                        ),
                        nn.PixelShuffle(2),
                    ]
                )
        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class EDSR_ARF(nn.Module):
    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        n_resblocks: int = 16,
        n_feats: int = 64,
        res_scale: float = 0.1,
        alpha: float = 2.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        t_min: float = 3.0,
        t_max: float = 11.0,
        kernel_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"EDSR_ARF currently supports scale=2/4, got {scale}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be > 0")
        if n_resblocks <= 0 or n_feats <= 0:
            raise ValueError("n_resblocks and n_feats must be > 0")

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        layer_channels = self._build_layer_channel_schedule(
            scale=scale,
            n_resblocks=n_resblocks,
            n_feats=n_feats,
            out_channels=out_channels,
        )
        num_conv_layers = len(layer_channels)

        if kernel_sizes is not None:
            if len(kernel_sizes) != num_conv_layers:
                raise ValueError(
                    f"EDSR_ARF needs {num_conv_layers} kernel sizes, got {len(kernel_sizes)}"
                )
            resolved_kernels = [int(k) for k in kernel_sizes]
        else:
            resolved_kernels = compute_adaptive_kernel_sizes(
                channels=layer_channels,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                t_min=t_min,
                t_max=t_max,
            )

        for k in resolved_kernels:
            if k <= 0 or k % 2 == 0:
                raise ValueError(
                    f"All kernel sizes must be positive odd integers, got: {resolved_kernels}"
                )

        self.kernel_sizes: Tuple[int, ...] = tuple(resolved_kernels)
        self.adaptive_params: Dict[str, object] = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "t_min": t_min,
            "t_max": t_max,
            "channels": tuple(layer_channels),
        }

        kernels_iter = iter(self.kernel_sizes)

        head_k = next(kernels_iter)
        self.head = nn.Conv2d(
            in_channels,
            n_feats,
            kernel_size=head_k,
            stride=1,
            padding=head_k // 2,
        )

        body_layers: List[nn.Module] = []
        for _ in range(n_resblocks):
            k1 = next(kernels_iter)
            k2 = next(kernels_iter)
            body_layers.append(
                AdaptiveResidualBlock(
                    n_feats=n_feats,
                    kernel_size_1=k1,
                    kernel_size_2=k2,
                    res_scale=res_scale,
                )
            )
        body_tail_k = next(kernels_iter)
        body_layers.append(
            nn.Conv2d(
                n_feats,
                n_feats,
                kernel_size=body_tail_k,
                stride=1,
                padding=body_tail_k // 2,
            )
        )
        self.body = nn.Sequential(*body_layers)

        upsample_kernel_count = 1 if scale == 2 else 2
        upsample_kernels = [next(kernels_iter) for _ in range(upsample_kernel_count)]

        tail_k = next(kernels_iter)
        self.tail = nn.Sequential(
            AdaptiveUpsampleBlock(scale=scale, n_feats=n_feats, kernel_sizes=upsample_kernels),
            nn.Conv2d(
                n_feats,
                out_channels,
                kernel_size=tail_k,
                stride=1,
                padding=tail_k // 2,
            ),
        )
        if any(True for _ in kernels_iter):
            raise RuntimeError(
                "Internal kernel schedule mismatch: unused kernel sizes remain."
            )

        self._initialize_weights()

    @staticmethod
    def _build_layer_channel_schedule(
        scale: int,
        n_resblocks: int,
        n_feats: int,
        out_channels: int,
    ) -> List[int]:
        channels: List[int] = [n_feats]
        for _ in range(n_resblocks):
            channels.extend([n_feats, n_feats])
        channels.append(n_feats)

        upsample_count = 1 if scale == 2 else 2
        channels.extend([n_feats * 4] * upsample_count)
        channels.append(out_channels)
        return channels

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extra_repr(self) -> str:
        preview = ", ".join(str(k) for k in self.kernel_sizes[:8])
        if len(self.kernel_sizes) > 8:
            preview = preview + ", ..."
        return f"scale={self.scale}, kernels=[{preview}] (n={len(self.kernel_sizes)})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"EDSR_ARF expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"EDSR_ARF expects {self.in_channels} channels, got {x.shape[1]}"
            )

        feat = self.head(x)
        res = self.body(feat) + feat
        return self.tail(res)


if __name__ == "__main__":
    model = EDSR_ARF(scale=4)
    print(model)
    print(f"Number of adaptive kernels: {len(model.kernel_sizes)}")
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    print("Output shape:", tuple(y.shape))
