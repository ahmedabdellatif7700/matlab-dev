import numpy as np
import torch
import torch.nn as nn


class VariableAlphaSigmoid(nn.Module):

    def __init__(self, alpha=4.0):
        super().__init__()

        self.alpha = nn.Parameter(
            torch.tensor(
                float(alpha),
                dtype=torch.float32,
            )
        )

    def forward(self, x):

        return torch.sigmoid(
            self.alpha * x
        )


def variable_alpha_sigmoid(
    x,
    alpha,
):

    return (
        1.0
        / (
            1.0
            + np.exp(-alpha * x)
        )
    )
