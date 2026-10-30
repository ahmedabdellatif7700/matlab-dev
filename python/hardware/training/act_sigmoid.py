"""h = sigmoid(z).  RTL: activation_sigmoid (ROM + one 32x22 multiply, no parameters)."""
import math

import numpy as np

from .activation import Activation, register
from .fixedpoint import ONE, qmul, round_shift

# Q8.24 sigmoid: 33-point table on [0, 8] (step 0.25), linear interpolation, odd symmetry.
# RTL: ROM + one 32x22 multiply.
SIG_SHIFT = 22
SIG_TABLE = tuple(int(round(ONE / (1.0 + math.exp(-k / 4.0)))) for k in range(33))
SIG_LIMIT = 8 * ONE


def sigmoid_q24(z):
    a = abs(z)
    if a >= SIG_LIMIT:
        y = SIG_TABLE[-1]
    else:
        i, frac = a >> SIG_SHIFT, a & ((1 << SIG_SHIFT) - 1)
        y = SIG_TABLE[i] + round_shift((SIG_TABLE[i + 1] - SIG_TABLE[i]) * frac, SIG_SHIFT)
    return y if z >= 0 else ONE - y


@register
class Sigmoid(Activation):
    """h = sigmoid(z)"""
    name = "sigmoid"

    def forward(self, z, p):
        return sigmoid_q24(z)

    def derivative(self, h, z, p):
        return qmul(h, ONE - h)

    def rom_words(self):
        return list(SIG_TABLE)

    def float_forward(self, z, p):
        return 1.0 / (1.0 + np.exp(-z))

    def float_derivative(self, z, p):
        h = self.float_forward(z, p)
        return h * (1.0 - h)
