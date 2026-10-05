"""Training objectives: L1 pixel loss plus an optional Sobel edge term.

    L = L1(SR, HR) + lambda * L_edge

    L_edge (static)   = mean |S(SR) - S(HR)|
    L_edge (adaptive) = mean  w * |S(SR) - S(HR)|,   w = minmax_per_image(S(HR))

S(.) is the per-channel Sobel gradient magnitude. The adaptive weight w lies in [0, 1]
and is computed from the ground truth only, so it carries no gradient. See
docs/METHOD.md for the derivation and docs/AUDIT.md for how this differs from the
original notebook (kernel typo, zero padding, batch-level normalisation).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

EDGE_MODES = ("none", "static", "adaptive")

_SOBEL_X = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]])
_SOBEL_Y = _SOBEL_X.t().contiguous()  # [[-1,-2,-1],[0,0,0],[1,2,1]]


def sobel_gradients(x: torch.Tensor):
    """Depthwise Sobel responses (gx, gy) of a [B, C, H, W] tensor.

    Replicate padding keeps the output size and, unlike zero padding, does not create a
    fake edge along the image border.
    """
    c = x.shape[1]
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype).expand(c, 1, 3, 3)
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype).expand(c, 1, 3, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(xp, kx, groups=c), F.conv2d(xp, ky, groups=c)


def sobel_magnitude(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """sqrt(gx^2 + gy^2 + eps). The eps keeps the gradient finite where gx = gy = 0."""
    gx, gy = sobel_gradients(x)
    return torch.sqrt(gx * gx + gy * gy + eps)


def adaptive_edge_weight(edge_target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalise the target edge map *per image* to [0, 1].

    The original notebook normalised over the whole batch, which made the weight of an
    image depend on which other images happened to share its batch.
    """
    b = edge_target.shape[0]
    flat = edge_target.reshape(b, -1)
    lo = flat.min(dim=1).values.view(b, 1, 1, 1)
    hi = flat.max(dim=1).values.view(b, 1, 1, 1)
    return ((edge_target - lo) / (hi - lo + eps)).detach()


@dataclass
class LossOutput:
    total: torch.Tensor
    pixel: torch.Tensor
    edge: torch.Tensor          # unweighted edge term (0 when edge_mode == "none")
    edge_weight_mean: torch.Tensor  # mean of w (1 for static, 0 for none); logged for analysis


class EdgeAwareLoss(nn.Module):
    """L1 pixel loss with an optional static or adaptive Sobel edge term."""

    def __init__(self, edge_mode: str = "none", edge_weight: float = 0.0):
        super().__init__()
        if edge_mode not in EDGE_MODES:
            raise ValueError(f"edge_mode must be one of {EDGE_MODES}, got {edge_mode!r}")
        if edge_mode == "none" and edge_weight != 0.0:
            raise ValueError("edge_weight must be 0 when edge_mode='none'")
        if edge_mode != "none" and edge_weight <= 0.0:
            raise ValueError("edge_weight must be > 0 for an edge loss")
        self.edge_mode = edge_mode
        self.edge_weight = float(edge_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        # Always evaluate the objective in fp32, also when the model ran under autocast.
        pred = pred.float()
        target = target.float()
        pixel = F.l1_loss(pred, target)
        if self.edge_mode == "none":
            zero = pixel.new_zeros(())
            return LossOutput(pixel, pixel, zero, zero)

        e_pred = sobel_magnitude(pred)
        with torch.no_grad():
            e_true = sobel_magnitude(target)
        diff = (e_pred - e_true).abs()
        if self.edge_mode == "static":
            edge = diff.mean()
            w_mean = pixel.new_ones(())
        else:
            w = adaptive_edge_weight(e_true)
            edge = (w * diff).mean()
            w_mean = w.mean()
        total = pixel + self.edge_weight * edge
        return LossOutput(total, pixel, edge, w_mean)


def build_loss(edge_mode: str, edge_weight: float) -> EdgeAwareLoss:
    return EdgeAwareLoss(edge_mode=edge_mode, edge_weight=edge_weight)
