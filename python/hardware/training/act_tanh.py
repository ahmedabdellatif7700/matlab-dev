"""Signed Q8.24 tanh using the supplied sigmoid ROM: 2*sigmoid(2*z)-1."""
import numpy as np
from .activation import Activation, register
from .fixedpoint import ONE, qmul
from .act_sigmoid import sigmoid_q24, SIG_TABLE


@register
class Tanh(Activation):
    name = "tanh"

    def forward(self, z, p):
        return 2*sigmoid_q24(2*z)-ONE

    def derivative(self, h, z, p):
        return max(0, ONE-qmul(h, h))

    def rom_words(self):
        return list(SIG_TABLE)

    def float_forward(self, z, p):
        return np.tanh(z)

    def float_derivative(self, z, p):
        return 1-np.tanh(z)**2
