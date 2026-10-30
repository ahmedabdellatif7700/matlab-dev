"""RAM word map. RTL counterpart: rtl/core/nn_params.vh (same word addresses)."""
from .fixedpoint import to_q

N0, N1, N2 = 4, 12, 2
RAM_WORDS = 256                                  # 8-bit address
W0, B0, W1, B1 = 0x00, 0x30, 0x3C, 0x54          # 48 + 12 + 24 + 2 parameters
A0, A1, A2, T = 0x56, 0x5A, 0x66, 0x68           # input, hidden, output, target
D1, D2 = 0x6A, 0x76                              # deltas
ACT_P, ACT_G, ACT_MAX_PARAMS = 0x78, 0x7C, 4     # activation parameters / their gradients
Z1 = 0x80                                        # 12 words: hidden pre-activation z

LAYERS = (
    dict(n_in=N0, n_out=N1, w=W0, b=B0, src=A0, dst=A1, delta=D1, hidden=True, z=Z1),
    dict(n_in=N1, n_out=N2, w=W1, b=B1, src=A1, dst=A2, delta=D2, hidden=False),
)


def new_ram():
    return [0] * RAM_WORDS


def load_parameters(ram, values):
    """Layer weights/biases in the same order as the supplied engine."""
    sizes = (N0*N1, N1, N1*N2, N2)
    if len(values) != sum(sizes):
        raise ValueError("parameter count disagrees with configured layout")
    offset = 0
    for base, size in zip((W0, B0, W1, B1), sizes):
        for i, value in enumerate(values[offset:offset+size]):
            ram[base+i] = to_q(float(value))
        offset += size


def configure(n_inputs=4, n_hidden=12):
    """Resize the same two-layer datapath; (4,12) preserves every RTL address.

    Call before constructing a model. Different layouts are run sequentially,
    not concurrently; this reference model has a single global RAM map.
    """
    global N0, N1, N2, W0, B0, W1, B1, A0, A1, A2, T, D1, D2
    global ACT_P, ACT_G, Z1, RAM_WORDS, LAYERS
    if n_inputs < 1 or n_hidden < 1:
        raise ValueError("positive input and hidden dimensions required")
    N0, N1, N2 = int(n_inputs), int(n_hidden), 2
    W0 = 0
    B0 = W0 + N0*N1
    W1 = B0 + N1
    B1 = W1 + N1*N2
    A0 = B1 + N2
    A1 = A0 + N0
    A2 = A1 + N1
    T = A2 + N2
    D1 = T + N2
    D2 = D1 + N1
    ACT_P = D2 + N2
    ACT_G = ACT_P + ACT_MAX_PARAMS
    Z1 = ACT_G + ACT_MAX_PARAMS
    RAM_WORDS = max(256, 1 << (Z1 + N1 - 1).bit_length())
    LAYERS = (
        dict(n_in=N0, n_out=N1, w=W0, b=B0, src=A0, dst=A1, delta=D1, hidden=True, z=Z1),
        dict(n_in=N1, n_out=N2, w=W1, b=B1, src=A1, dst=A2, delta=D2, hidden=False),
    )
