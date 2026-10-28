import numpy as np
import torch
import torch.nn as nn


class StableAlphaSigmoid(nn.Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):

        return torch.sigmoid(
            self.alpha * x
        )


def stable_alpha_sigmoid(
    x,
    alpha=1.0,
):

    return (
        1.0
        / (
            1.0
            + np.exp(-alpha * x)
        )
    )
