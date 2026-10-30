"""forward_atomic(layer): out[j] = act(sat32(round(bias*2^24 + sum w*x))).  RTL: forward_atomic."""
from .fixedpoint import FRAC, round_shift, sat32
from . import memory_map as mm


def forward_atomic(ram, layer, act):
    """out[j] = act(sat32(round(bias*2^24 + sum w*x))); hidden layer also stores z; output is linear."""
    c = mm.LAYERS[layer]
    p = tuple(ram[mm.ACT_P:mm.ACT_P + act.n_params])
    for j in range(c["n_out"]):
        acc = ram[c["b"] + j] << FRAC
        for i in range(c["n_in"]):
            acc += ram[c["src"] + i] * ram[c["w"] + j * c["n_in"] + i]
        z = sat32(round_shift(acc, FRAC), "z")
        if c["hidden"]:
            ram[c["z"] + j] = z
            ram[c["dst"] + j] = act.forward(z, p)
        else:
            ram[c["dst"] + j] = z
