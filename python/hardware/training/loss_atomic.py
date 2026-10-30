"""loss_atomic: d2 = a2 - t.  RTL: loss_atomic."""
from .fixedpoint import sat32
from . import memory_map as mm


def loss_atomic(ram):
    """d2 = a2 - t, the MSELoss gradient at the linear output (mean over 2 outputs, factor 2 cancels)."""
    for k in range(mm.N2):
        ram[mm.D2 + k] = sat32(ram[mm.A2 + k] - ram[mm.T + k], "loss")
