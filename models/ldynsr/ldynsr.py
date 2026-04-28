import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import conv3x3, init_kaiming
from .dyna import DynA
from .frm import FRM


class LDynSR(nn.Module):
    """Lightweight Dynamic Attention Network for infrared SR."""

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        feat_channels: int = 48,
        num_dyna: int = 6,
        dam_reduction: int = 16,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"LDynSR currently supports scale=2/4, got {scale}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be > 0")
        if feat_channels <= 0 or num_dyna <= 0:
            raise ValueError("feat_channels and num_dyna must be > 0")
        if dam_reduction <= 0:
            raise ValueError("dam_reduction must be > 0")

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.num_dyna = num_dyna
        self.dam_reduction = dam_reduction

        self.shallow = conv3x3(in_channels, feat_channels)
        self.dfeb_blocks = nn.Sequential(
            *[DynA(feat_channels, dam_reduction=dam_reduction) for _ in range(num_dyna)]
        )
        self.dfeb_tail = conv3x3(feat_channels, feat_channels)
        self.frm = FRM(channels=feat_channels, scale=scale)
        self.recon = conv3x3(feat_channels, out_channels)

        init_kaiming(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"LDynSR expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"LDynSR expects {self.in_channels} channels, got {x.shape[1]}"
            )

        shallow = self.shallow(x)
        deep = self.dfeb_blocks(shallow)
        deep = self.dfeb_tail(deep) + shallow
        sr_feat = self.frm(deep)
        sr_res = self.recon(sr_feat)

        bicubic = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        return sr_res + bicubic
