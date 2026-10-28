"""Smooth multilevel activation and its trainable-shape ablation."""
import math
import torch
from torch import nn

class Multilevel(nn.Module):
    """Shared smooth staircase; learned widths/locations change its shape.

    This is an experimental activation ablation inspired by multilevel
    sigmoids, not a claim of mathematical or literature novelty.
    """
    def __init__(self, learned=False):
        super().__init__()
        self.learned = learned
        centers = torch.tensor([-1.0, 0.0, 1.0])
        slopes = torch.full((3,), math.log(math.expm1(2.0)))
        if learned:
            self.centers = nn.Parameter(centers)
            self.raw_slopes = nn.Parameter(slopes)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('raw_slopes', slopes)

    def forward(self, x):
        slope = nn.functional.softplus(self.raw_slopes)+1e-4
        return (2*torch.sigmoid((x[..., None]-self.centers)*slope).sum(dim=-1)/3)-1


