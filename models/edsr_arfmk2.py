"""EDSR with Adaptive Receptive Field MK2 (multi-branch soft routing).

Implements Section 3 of ``Adaptive Reception Field.md``:
instead of predicting a single quantised kernel size per layer (MK1),
each body layer runs **multiple parallel branches** with different
effective receptive fields and combines them via a soft Softmax-weighted
sum.

Branch set R (Section 7, "minimum viable version"):

    branch 0 : DW-3×3, dilation=1   (small, local context)
    branch 1 : DW-3×3, dilation=2   (medium context via dilation)
    branch 2 : DW-5×5, dilation=1   (larger local context)

Routing weight for layer i:

    w_i = Softmax( MLP( [d_i, c_i, d_i * c_i] ) )

where [d_i, c_i] are the same structural features as MK1 (Section 2):

    d_i = i / (L - 1)                           # relative depth ∈ [0,1]
    c_i = (log C_i - log C_min)
          / (log C_max - log C_min)              # log-channel ∈ [0,1]

A **single MLP is shared across all layers** so that structurally
similar positions receive similar routing weights (weight sharing /
inductive bias).  The MLP is optimised end-to-end with the rest of the
network.

Output for layer i:

    y_i = PW( Σ_m  w_i^(m) · DW_m(x_i) )

where DW_m are depthwise convolutions (spatial mixing) and PW is a
shared pointwise 1×1 conv (channel mixing).  This decomposition keeps
per-branch parameter count small while preserving expressiveness.

Relation to MK1 (hard-quantised single kernel):
  * No hard quantisation – soft, differentiable routing.
  * Naturally fits the "parallel multi-kernel" idiom of MRIDN / InfraFFN.
  * Routing weights can be inspected post-training to visualise which
    structural positions prefer which receptive field.

Head, upsample, and output convolutions remain standard Conv2d to match
the original EDSR reconstruction structure.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Structural feature computation  (identical formulas to MK1 / Section 2)
# ---------------------------------------------------------------------------

def _compute_structural_features(
    channels: Sequence[int],
) -> List[Tuple[float, float]]:
    """Return per-layer (d_i, c_i) from the ARF structural formulas.

    d_i = i / (L - 1)  for i in [0, L-1]  — relative depth ∈ [0, 1]
    c_i = (log C_i - log C_min) / (log C_max - log C_min) — log-channel ∈ [0, 1]

    When all channels are equal (as in the EDSR body), c_i collapses to
    0.5 for every layer.  The router MLP will then learn purely depth-based
    routing, which is still a meaningful and useful structural prior.
    """
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

    return list(zip(d_list, c_list))


# ---------------------------------------------------------------------------
# Shared MLP branch router
# ---------------------------------------------------------------------------

class ARFBranchRouter(nn.Module):
    """Shared MLP: structural features → branch weights.

    The same router instance is reused by every ARFBranchLayer in the
    model body.  Sharing means:

      * Structurally similar layers (similar d_i, c_i) get similar weights.
      * Total router parameter count is tiny (3→hidden→num_branches).

    Input: [d_i, c_i, d_i * c_i]   (3-D fixed scalar vector per layer)
    Output: softmax-normalised branch weights  (num_branches-D)

    Args:
        num_branches: number of parallel depthwise branches (default 3).
        hidden_dim:   MLP hidden width (default 32).
    """

    def __init__(self, num_branches: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_branches),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, struct_feat: torch.Tensor) -> torch.Tensor:
        """Compute softmax branch weights from the layer's structural features.

        Args:
            struct_feat: (3,) float tensor — [d_i, c_i, d_i * c_i].
                         Stored as a non-trainable buffer in ARFBranchLayer
                         so it automatically moves with .to(device).
        Returns:
            weights: (num_branches,) tensor, sums to 1.
        """
        logits = self.mlp(struct_feat.unsqueeze(0))   # (1, num_branches)
        return torch.softmax(logits, dim=-1).squeeze(0)  # (num_branches,)


# ---------------------------------------------------------------------------
# Branch specifications  (Section 7)
# ---------------------------------------------------------------------------

# Each entry is (kernel_size, dilation).  Three branches as recommended.
_BRANCH_SPECS: Tuple[Tuple[int, int], ...] = (
    (3, 1),   # DW-3×3-d1  — local detail
    (3, 2),   # DW-3×3-d2  — medium context (effective 5×5)
    (5, 1),   # DW-5×5-d1  — larger local context
)


# ---------------------------------------------------------------------------
# Multi-branch spatial mixing layer
# ---------------------------------------------------------------------------

class ARFBranchLayer(nn.Module):
    """One spatial mixing layer: parallel DWConv branches + MLP routing + PW mix.

    Forward pass:

        weights = router([d_i, c_i, d_i*c_i])          # (num_branches,)
        branch_outs = stack([DW_m(x) for m in branches])  # (M, B, C, H, W)
        mixed = (branch_outs * weights[:, None, None, None, None]).sum(0)
        y = pointwise(mixed)                            # channel mixing

    The structural feature vector is fixed at construction time and stored
    as a registered buffer so it moves to the correct device automatically.

    Args:
        n_feats:  number of feature channels (in and out).
        d_i:      relative depth of this layer in [0, 1].
        c_i:      log-normalised channel width in [0, 1].
        router:   shared ARFBranchRouter instance (not owned by this layer).
    """

    def __init__(
        self,
        n_feats: int,
        d_i: float,
        c_i: float,
        router: ARFBranchRouter,
    ) -> None:
        super().__init__()
        self.router = router

        # Fixed structural features stored as buffer for device portability.
        self.register_buffer(
            "struct_feat",
            torch.tensor([d_i, c_i, d_i * c_i], dtype=torch.float32),
        )

        # Depthwise branches — spatial mixing only (groups = n_feats).
        dw_branches: List[nn.Module] = []
        for k, d in _BRANCH_SPECS:
            pad = (k // 2) * d  # same-spatial-size padding with dilation
            dw_branches.append(
                nn.Conv2d(
                    n_feats, n_feats,
                    kernel_size=k,
                    stride=1,
                    padding=pad,
                    dilation=d,
                    groups=n_feats,   # depthwise: no cross-channel mixing here
                    bias=False,
                )
            )
        self.dw_branches = nn.ModuleList(dw_branches)

        # Pointwise 1×1 conv for channel mixing after the weighted sum.
        self.pointwise = nn.Conv2d(n_feats, n_feats, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get soft branch weights from the shared router.
        weights = self.router(self.struct_feat)   # (num_branches,)

        # Run all branches and form a (M, B, C, H, W) stack.
        branch_outs = torch.stack(
            [branch(x) for branch in self.dw_branches], dim=0
        )   # (M, B, C, H, W)

        # Weighted sum across the branch dimension.
        w = weights.view(len(self.dw_branches), 1, 1, 1, 1)
        mixed = (branch_outs * w).sum(dim=0)   # (B, C, H, W)

        return self.pointwise(mixed)


# ---------------------------------------------------------------------------
# Residual block with two ARFBranchLayer instances
# ---------------------------------------------------------------------------

class ARFMk2ResidualBlock(nn.Module):
    """EDSR-style residual block with both spatial convolutions replaced by ARFBranchLayer.

    Each of the two positions has its own (d_i, c_i) structural features
    but shares the same ARFBranchRouter with every other layer in the model.

    Args:
        n_feats:           feature channel count.
        d_i1, c_i1:       structural features for the first conv position.
        d_i2, c_i2:       structural features for the second conv position.
        router:            shared ARFBranchRouter instance.
        res_scale:         residual scaling factor (0.1 follows EDSR paper).
    """

    def __init__(
        self,
        n_feats: int,
        d_i1: float,
        c_i1: float,
        d_i2: float,
        c_i2: float,
        router: ARFBranchRouter,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ARFBranchLayer(n_feats, d_i1, c_i1, router),
            nn.ReLU(inplace=True),
            ARFBranchLayer(n_feats, d_i2, c_i2, router),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


# ---------------------------------------------------------------------------
# Upsample block (unchanged from edsr.py)
# ---------------------------------------------------------------------------

class UpsampleBlock(nn.Module):
    def __init__(self, scale: int, n_feats: int) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"EDSR_ARFMk2 supports scale=2/4, got {scale}")
        layers: List[nn.Module] = []
        if scale == 2:
            layers.extend([
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
            ])
        else:
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                ])
        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class EDSR_ARFMk2(nn.Module):
    """EDSR with Adaptive Receptive Field MK2 (multi-branch soft routing).

    The body's spatial mixing (both conv layers in every ResBlock, plus the
    body-tail conv) is replaced with parallel depthwise branches routed by a
    shared MLP whose inputs are fixed structural features [d_i, c_i, d_i*c_i].
    Head, upsample, and output convolutions remain standard Conv2d.

    Structural feature assignment:
      * Only the ``n_arf_layers = 2 * n_resblocks + 1`` body layers participate
        in the ARF schedule (ResBlock conv 1, conv 2, …, body-tail conv).
      * All EDSR body layers have the same channel count (n_feats), so c_i
        collapses to 0.5.  The router learns purely depth-based routing; this
        is still a meaningful structural prior (early layers prefer small RF,
        late layers prefer large RF).

    Args:
        scale:             SR upscale factor (2 or 4).
        in_channels:       input image channels (1 for infrared grayscale).
        out_channels:      output image channels.
        n_resblocks:       number of residual blocks.
        n_feats:           base feature channel width.
        res_scale:         residual scaling factor.
        router_hidden_dim: hidden width of the shared MLP router (default 32).
    """

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        n_resblocks: int = 16,
        n_feats: int = 64,
        res_scale: float = 0.1,
        router_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"EDSR_ARFMk2 supports scale=2/4, got {scale}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be > 0")
        if n_resblocks <= 0 or n_feats <= 0:
            raise ValueError("n_resblocks and n_feats must be > 0")

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- Shared router (one MLP for the whole model) ---
        self.router = ARFBranchRouter(
            num_branches=len(_BRANCH_SPECS),
            hidden_dim=router_hidden_dim,
        )

        # --- Structural features for all ARF body layers ---
        # Layout: [ResBlock_0_conv1, ResBlock_0_conv2, ..., body_tail]
        # Total: 2*n_resblocks + 1 layers, all with n_feats channels.
        n_arf_layers = 2 * n_resblocks + 1
        arf_channels = [n_feats] * n_arf_layers
        struct_feats = _compute_structural_features(arf_channels)
        feat_iter = iter(struct_feats)

        # --- Head (standard Conv2d — feature extraction) ---
        self.head = nn.Conv2d(
            in_channels, n_feats, kernel_size=3, stride=1, padding=1
        )

        # --- Body: residual blocks + body-tail ARFBranchLayer ---
        body_layers: List[nn.Module] = []
        for _ in range(n_resblocks):
            d1, c1 = next(feat_iter)
            d2, c2 = next(feat_iter)
            body_layers.append(
                ARFMk2ResidualBlock(
                    n_feats=n_feats,
                    d_i1=d1, c_i1=c1,
                    d_i2=d2, c_i2=c2,
                    router=self.router,
                    res_scale=res_scale,
                )
            )
        # Body-tail: deepest ARF position
        d_t, c_t = next(feat_iter)
        body_layers.append(ARFBranchLayer(n_feats, d_t, c_t, self.router))
        self.body = nn.Sequential(*body_layers)

        # Sanity-check that all structural features were consumed
        remaining = sum(1 for _ in feat_iter)
        if remaining != 0:
            raise RuntimeError(
                f"Internal error: {remaining} structural feature(s) unused "
                f"after building the body."
            )

        # --- Tail: upsample + output conv (standard) ---
        self.tail = nn.Sequential(
            UpsampleBlock(scale=scale, n_feats=n_feats),
            nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _initialize_weights(self) -> None:
        """Kaiming-normal for Conv2d; router Linear layers keep xavier from init."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def branch_specs(self) -> Tuple[Tuple[int, int], ...]:
        """Return the (kernel, dilation) specs of the spatial branches."""
        return _BRANCH_SPECS

    def get_routing_weights(self) -> Dict[str, torch.Tensor]:
        """Return the current routing weights for every ARFBranchLayer.

        Keys are formatted as ``"body.<module_index>.<sub_index>"`` to
        mirror the nn.Sequential path.  Useful for inspection / logging.

        Returns:
            dict mapping layer path → (num_branches,) weight tensor.
        """
        result: Dict[str, torch.Tensor] = {}
        self.router.eval()
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, ARFBranchLayer):
                    w = self.router(module.struct_feat)
                    result[name] = w
        return result

    def extra_repr(self) -> str:
        return (
            f"scale={self.scale}, "
            f"branches={list(_BRANCH_SPECS)}, "
            f"router_hidden={self.router.mlp[0].out_features}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"EDSR_ARFMk2 expects [B, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"EDSR_ARFMk2 expects {self.in_channels} input channel(s), "
                f"got {x.shape[1]}"
            )
        feat = self.head(x)
        res = self.body(feat) + feat
        return self.tail(res)


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = EDSR_ARFMk2(scale=2)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters : {total_params:,}")

    print("\nPer-layer routing weights (post-init):")
    for path, w in model.get_routing_weights().items():
        specs = [f"DW{k}×{k}-d{d}" for k, d in _BRANCH_SPECS]
        weight_str = "  ".join(f"{s}={v:.3f}" for s, v in zip(specs, w.tolist()))
        print(f"  {path:50s}  {weight_str}")

    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    print(f"\nInput : {tuple(x.shape)}  →  Output : {tuple(y.shape)}")
