"""update_atomic(layer): SGD step on weights, biases and trainable activation parameters.  RTL: update_atomic."""
from .fixedpoint import qmul, sat32
from . import memory_map as mm

LR_Q24 = 16777          # 0.001 in Q8.24, hardcoded for now (later a PS register)


def update_atomic(ram, layer, act, lr_q24=LR_Q24):
    """SGD: new = sat32(old - qmul(g, lr)); g(weight) = qmul(delta[j], input[i]), g(bias) = delta[j].
    The hidden layer's update also steps the trainable activation parameters."""
    c = mm.LAYERS[layer]
    for j in range(c["n_out"]):
        d = ram[c["delta"] + j]
        for i in range(c["n_in"]):
            a = c["w"] + j * c["n_in"] + i
            ram[a] = sat32(ram[a] - qmul(qmul(d, ram[c["src"] + i], "grad"), lr_q24), "master")
        a = c["b"] + j
        ram[a] = sat32(ram[a] - qmul(d, lr_q24), "master")
    if layer == 0:
        for k in act.trainable:
            ram[mm.ACT_P + k] = sat32(ram[mm.ACT_P + k] - qmul(ram[mm.ACT_G + k], lr_q24), "master")
