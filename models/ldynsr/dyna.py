import torch
import torch.nn as nn

from .bta import BTA
from .common import conv1x1, conv3x3
from .dam import DAM
from .pa import PA


class DynA(nn.Module):
    """Dynamic Attention block composed of PA/BTA/NOA branches + DAM."""

    def __init__(self, channels: int, dam_reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")

        self.pre = conv1x1(channels, channels)
        self.pa_branch = PA(channels)
        self.bta_branch = BTA(channels, reduction=dam_reduction)
        # Figure (a): NOA branch is a plain 3x3 conv.
        self.noa_branch = conv3x3(channels, channels)
        self.dam = DAM(channels=channels, reduction=dam_reduction, num_branches=3)
        self.post = conv1x1(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.pre(x)
        pa_out = self.pa_branch(base)
        bta_out = self.bta_branch(base)
        noa_out = self.noa_branch(base)

        # DAM receives the block input per figure (a), not post-1x1 features.
        weights = self.dam(x)
        w_noa = weights[:, 0:1]
        w_pa = weights[:, 1:2]
        w_bta = weights[:, 2:3]
        fused = (
            noa_out * w_noa
            + pa_out * w_pa
            + bta_out * w_bta
        )
        fused = self.post(fused)
        return x + fused
