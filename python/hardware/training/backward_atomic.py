"""backward_atomic(layer): hidden delta and activation-parameter gradients.  RTL: backward_atomic
(+ the derivative / gradient outputs of activation_unit)."""
from .fixedpoint import FRAC, qmul, round_shift, sat32
from . import memory_map as mm


def backward_atomic(ram, layer, act):
    """Propagate the delta of `layer` to its input side (layer >= 1; layer 0 has no input delta).

    d_prev[j] = qmul( sat32(round(sum_k W[k][j] * d[k])), act.derivative(h[j], z[j], p) )
    Trainable activation parameters: RAM[mm.ACT_G+k] = sat32(round(sum_j act.param_grad(...)[k])).
    Reads the weights before the update phase changes them."""
    if layer == 0:
        return
    c, prev = mm.LAYERS[layer], mm.LAYERS[layer - 1]
    p = tuple(ram[mm.ACT_P:mm.ACT_P + act.n_params])
    gacc = [0] * len(act.trainable)
    for j in range(c["n_in"]):
        s = 0
        for k in range(c["n_out"]):
            s += ram[c["w"] + k * c["n_in"] + j] * ram[c["delta"] + k]
        back = sat32(round_shift(s, FRAC), "back")
        h, z = ram[c["src"] + j], ram[prev["z"] + j]
        ram[prev["delta"] + j] = qmul(back, act.derivative(h, z, p), "delta")
        for n, contribution in enumerate(act.param_grad(back, h, z, p)):
            gacc[n] += contribution
    for n, k in enumerate(act.trainable):
        ram[mm.ACT_G + k] = sat32(round_shift(gacc[n], FRAC), "galpha")
