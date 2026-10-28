import numpy as np
import torch
import torch.nn as nn


class MultilevelSigmoid(nn.Module):

    def __init__(self, c1=6.0, levels=4):
        super().__init__()

        self.c1 = float(c1)
        self.levels = int(levels)

    def forward(self, x):

        exp_neg = torch.exp(
            x.new_tensor(-self.c1)
        )

        c2 = (
            (1.0 + exp_neg)
            / (1.0 - exp_neg)
        )

        max_level = self.levels // 2 - 1

        level = torch.ceil(
            (x - 1.0) / 2.0
        )

        level = torch.clamp(
            level,
            -max_level,
            max_level,
        )

        return (
            2.0
            * c2
            / (
                1.0
                + torch.exp(
                    -self.c1
                    * (x - 2.0 * level)
                )
            )
            - c2
            + 2.0 * level
        )


class ComplexMultilevelSigmoid(nn.Module):

    def __init__(self, c1=6.0, levels=4):
        super().__init__()

        self.activation = MultilevelSigmoid(
            c1=c1,
            levels=levels,
        )

    def forward(
        self,
        real,
        imag,
    ):

        return (
            self.activation(real),
            self.activation(imag),
        )


def multilevel_sigmoid(
    x,
    c1=6.0,
    levels=4,
):

    exp_neg = np.exp(-c1)

    c2 = (
        (1.0 + exp_neg)
        / (1.0 - exp_neg)
    )

    max_level = levels // 2 - 1

    level = np.ceil(
        (x - 1.0) / 2.0
    )

    level = np.clip(
        level,
        -max_level,
        max_level,
    )

    return (
        2.0
        * c2
        / (
            1.0
            + np.exp(
                -c1
                * (x - 2.0 * level)
            )
        )
        - c2
        + 2.0 * level
    )


def complex_multilevel_sigmoid(
    x,
    c1=6.0,
    levels=4,
):

    return (
        multilevel_sigmoid(
            np.real(x),
            c1=c1,
            levels=levels,
        )
        + 1j
        * multilevel_sigmoid(
            np.imag(x),
            c1=c1,
            levels=levels,
        )
    )