"""SRCNN with Adaptive Receptive Field (ARF) kernel sizes.

This is a sibling of ``srcnn.py``. The classical SRCNN script is left
untouched so the two variants can be trained and compared side by side.

The only structural difference vs. the classical SRCNN is that the kernel
sizes of the three conv layers are NOT hard-coded to (9, 5, 5). Instead
they are derived at construction time from the explicit formula in
Section 2 ("最基础的显式公式") of *Adaptive Receptive Field*:

    d_i = (i - 1) / (L - 1)                          # relative depth
    c_i = (log C_i - log C_min)
          / (log C_max - log C_min)                  # log channel scale
    s_i = alpha * d_i + beta * c_i + gamma * d_i * c_i
    t_i = t_min + (t_max - t_min) * sigmoid(s_i)
    k_i = Q_odd(t_i)                                  # nearest odd in {3,5,7,9,11}

The default weights (alpha:beta:gamma = 2:1:1) follow the recommendation
in Section 4 of the document: depth dominates, channel modulates, and the
interaction term pushes wide+deep layers further up.
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Adaptive Receptive Field formula (Section 2)
# ----------------------------------------------------------------------------

def _nearest_odd(x: float, k_min: int = 3, k_max: int = 11) -> int:
    """Quantize a positive real value to the nearest odd integer in [k_min, k_max].

    Implements ``Q_odd(t_i)`` from Section 2:
        \\hat t_i = Q_odd(t_i),  \\hat t_i \\in {3, 5, 7, 9, 11}

    Even-rounding ties are broken toward the side ``x`` actually lies on, so
    e.g. 10.24 -> 11 (closer to 11 than to 9), 9.6 -> 9, 7.4 -> 7.
    """
    k = int(round(x))
    if k % 2 == 0:
        k = k + 1 if x >= k else k - 1
    k = max(k_min, min(k_max, k))
    if k % 2 == 0:  # safety net if k_min/k_max were given as even numbers
        k += 1
    return k


def compute_adaptive_kernel_sizes(
    channels: Sequence[int],
    alpha: float = 2.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    t_min: float = 3.0,
    t_max: float = 11.0,
) -> List[int]:
    """Per-layer kernel sizes from the depth/channel joint formula.

    Args:
        channels: per-layer "operating width" C_i. For SRCNN we feed the
            output channel count of each conv layer, treating C_i as a
            capacity proxy as discussed in the document.
        alpha, beta, gamma: weights for depth, channel, and depth*channel
            interaction terms. Default 2:1:1 follows Section 4.
        t_min, t_max: continuous receptive-field bounds before quantization.

    Returns:
        A list of odd integers, one per layer, suitable as conv kernel sizes.
    """
    L = len(channels)
    if L == 0:
        return []

    # --- d_i: relative depth -------------------------------------------------
    if L == 1:
        d_list = [0.5]
    else:
        d_list = [i / (L - 1) for i in range(L)]

    # --- c_i: log-normalized channel scale -----------------------------------
    # The doc explicitly recommends log over linear because the marginal gain
    # of going 16->32 is not the same as 128->144.
    log_c = [math.log(max(int(c), 1)) for c in channels]
    log_min = min(log_c)
    log_max = max(log_c)
    denom = log_max - log_min
    if denom < 1e-8:
        c_list = [0.5] * L
    else:
        c_list = [(lc - log_min) / denom for lc in log_c]

    # --- s_i -> t_i -> k_i ---------------------------------------------------
    k_min = _nearest_odd(t_min, k_min=1, k_max=99)
    k_max = _nearest_odd(t_max, k_min=1, k_max=99)
    if k_max < k_min:
        k_min, k_max = k_max, k_min

    kernels: List[int] = []
    for d_i, c_i in zip(d_list, c_list):
        s_i = alpha * d_i + beta * c_i + gamma * d_i * c_i
        sigma_s = 1.0 / (1.0 + math.exp(-s_i))
        t_i = t_min + (t_max - t_min) * sigma_s
        kernels.append(_nearest_odd(t_i, k_min=k_min, k_max=k_max))
    return kernels


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class SRCNN_ARF(nn.Module):
    """SRCNN with Adaptive Receptive Field kernel sizes.

    Input must already be bicubic-upsampled to target size, exactly like the
    classical SRCNN. The forward pass and weight init are intentionally kept
    identical to ``srcnn.SRCNN`` so that any difference in training results
    can be attributed solely to the kernel-size schedule.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_features_1: feature width after conv1.
        num_features_2: feature width after conv2.
        alpha, beta, gamma: weights for depth / channel / interaction terms
            in the s_i formula. Default 2:1:1.
        t_min, t_max: continuous receptive-field bounds before quantization.
        kernel_sizes: optional manual override of the three kernel sizes
            (must be 3 odd ints). When provided, the formula is bypassed,
            which is useful for ablation against fixed configurations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features_1: int = 64,
        num_features_2: int = 32,
        alpha: float = 2.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        t_min: float = 3.0,
        t_max: float = 11.0,
        kernel_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Per-layer "operating width" C_i: the output channel count produced
        # by each conv layer, used as the capacity proxy in the doc.
        layer_channels: Tuple[int, int, int] = (
            num_features_1,
            num_features_2,
            out_channels,
        )

        if kernel_sizes is not None:
            if len(kernel_sizes) != 3:
                raise ValueError(
                    f"SRCNN_ARF expects exactly 3 kernel sizes, got {len(kernel_sizes)}"
                )
            k1, k2, k3 = (int(k) for k in kernel_sizes)
        else:
            k1, k2, k3 = compute_adaptive_kernel_sizes(
                channels=layer_channels,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                t_min=t_min,
                t_max=t_max,
            )

        for k in (k1, k2, k3):
            if k % 2 == 0 or k < 1:
                raise ValueError(
                    f"SRCNN_ARF kernel sizes must be positive odd integers, "
                    f"got {(k1, k2, k3)}"
                )

        # Expose the resolved schedule so training scripts / loggers can
        # easily report which receptive-field configuration was used.
        self.kernel_sizes: Tuple[int, int, int] = (k1, k2, k3)
        self.adaptive_params = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "t_min": t_min,
            "t_max": t_max,
            "channels": layer_channels,
        }

        # Build the three conv layers with the resolved kernel sizes.
        # padding = k // 2 keeps spatial size unchanged (stride=1).
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_features_1,
            kernel_size=k1,
            stride=1,
            padding=k1 // 2,
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=num_features_1,
            out_channels=num_features_2,
            kernel_size=k2,
            stride=1,
            padding=k2 // 2,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels=num_features_2,
            out_channels=out_channels,
            kernel_size=k3,
            stride=1,
            padding=k3 // 2,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extra_repr(self) -> str:
        return f"kernel_sizes={self.kernel_sizes}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"SRCNN_ARF expects 4D input [B, C, H, W], but got shape={tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"SRCNN_ARF expects input channels={self.in_channels}, "
                f"but got {x.shape[1]}"
            )

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


if __name__ == "__main__":
    # Quick sanity check: print the kernel schedule for the default config.
    model = SRCNN_ARF()
    print(model)
    print("Resolved kernel sizes:", model.kernel_sizes)
    x = torch.randn(1, 1, 64, 64)
    y = model(x)
    print("Output shape:", tuple(y.shape))
