"""Activation interface: the single contract between the training atomics and ANY hidden activation.

RTL counterpart: the port list of `activation_unit`. The training engine has ONE activation slot;
which function sits in it is chosen at build time, not at run time:

    activation_unit  (wrapper, parameter ACT_SEL)  --selects-->  activation_sigmoid   (ACT_SEL = 0, the only one)
    a further implementation would expose the same ports, so swapping = one parameter / module reference.

Ports (all signed Q8.24, 32 bit):
    in   z                       hidden pre-activation (from forward_atomic)
    in   p[0 .. n_params-1]      activation parameters, read from RAM[ACT_P + k]
    out  h       = forward(z, p)                       -> a1
    out  dh_dz   = derivative(h, z, p)                 -> hidden delta (backward_atomic)
    out  dparam_k = param_grad(back, h, z, p)          -> RAM[ACT_G + k] accumulators (trainable only)
  update_atomic steps each trainable parameter:  p[k] -= lr * RAM[ACT_G + k].
  An implementation declares n_params (<= ACT_MAX_PARAMS = 4), which parameters are trainable, and
  its ROM contents. Unused ports are tied off in RTL. Cycle-level timing (valid/ready) is fixed in
  the spec; this model is untimed.

param_grad returns the UNROUNDED per-unit contribution (Q8.48); backward_atomic sums it in a 64-bit
accumulator and rounds once. float_* methods are the float64 definition (comparison replay only).

Adding an activation = one new act_<name>.py with a subclass decorated with @register, plus one
import line at the bottom of this file. Example (hard sigmoid):

    @register
    class HardSigmoid(Activation):
        name = "hard_sigmoid"
        def forward(self, z, p):          return max(0, min(ONE, (z >> 2) + ONE // 2))
        def derivative(self, h, z, p):    return ONE // 4 if 0 < h < ONE else 0
        def float_forward(self, z, p):    return np.clip(0.25 * z + 0.5, 0, 1)
        def float_derivative(self, z, p): return np.where(abs(z) < 2, 0.25, 0.0)
"""
import numpy as np  # noqa: F401  (used by implementations)

from .fixedpoint import ONE, to_q  # noqa: F401

class Activation:
    name = "base"
    n_params = 0            # parameter words kept in RAM
    trainable = ()          # indices of the parameters trained by SGD

    def init_params(self):                      # Q8.24 words for RAM[ACT_P ...]
        return []

    def forward(self, z, p):                    # integer datapath
        raise NotImplementedError

    def derivative(self, h, z, p):              # dh/dz in Q8.24
        raise NotImplementedError

    def param_grad(self, back, h, z, p):        # one Q8.48 entry per trainable parameter
        return []

    def rom_words(self):                        # ROM contents for the RTL (none by default)
        return []

    def float_params(self):                     # float64 definition (replay only)
        return [w / ONE for w in self.init_params()]

    def float_forward(self, z, p):
        raise NotImplementedError

    def float_derivative(self, z, p):
        raise NotImplementedError

    def float_param_grad(self, back, z, p):
        return []


ACTIVATIONS = {}          # name -> class, filled by @register


def register(cls):
    ACTIVATIONS[cls.name] = cls
    return cls


def create_activation(name):
    """Factory used by the wrapper."""
    if name not in ACTIVATIONS:
        raise ValueError(f"unsupported activation {name!r}; choose from {tuple(ACTIVATIONS)}")
    cls = ACTIVATIONS[name]
    return cls()


# implementations (each one is the model of one RTL module behind the activation_unit ports)
from . import act_sigmoid  # noqa: E402,F401

from . import act_tanh  # noqa: E402,F401
