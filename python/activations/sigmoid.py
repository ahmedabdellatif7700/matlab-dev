import numpy as np
import torch
import torch.nn as nn


class Sigmoid(nn.Module):

    def forward(self, x):
        return torch.sigmoid(x)


def sigmoid(x):

    return 1.0 / (
        1.0 + np.exp(-x)
    )
