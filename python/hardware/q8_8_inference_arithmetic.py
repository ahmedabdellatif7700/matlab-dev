import numpy as np


TOTAL_BITS = 16
FRAC_BITS = 8
ACC_BITS = 40


def saturate(values, bits=TOTAL_BITS):

    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1

    return np.clip(
        values,
        minimum,
        maximum,
    ).astype(np.int64)


def quantize_fixed(values):

    scale = 1 << FRAC_BITS

    fixed = np.rint(
        np.asarray(values, dtype=np.float64) * scale
    ).astype(np.int64)

    return saturate(fixed)


def dequantize_fixed(values):

    return (
        np.asarray(values, dtype=np.float32)
        / float(1 << FRAC_BITS)
    )


def round_shift(values):

    values = np.asarray(values, dtype=np.int64)
    half = 1 << (FRAC_BITS - 1)

    magnitude = (
        np.abs(values) + half
    ) >> FRAC_BITS

    return np.where(
        values < 0,
        -magnitude,
        magnitude,
    )


def mac_q8_8(
    inputs,
    weights,
    biases,
):

    inputs = np.asarray(
        inputs,
        dtype=np.int64,
    )

    weights = np.asarray(
        weights,
        dtype=np.int64,
    )

    biases = np.asarray(
        biases,
        dtype=np.int64,
    )

    accumulator = (
        inputs @ weights.T
        + (biases << FRAC_BITS)
    )

    accumulator = saturate(
        accumulator,
        bits=ACC_BITS,
    )

    return saturate(
        round_shift(accumulator)
    )


def sigmoid_q8_8(values):

    values = np.asarray(
        values,
        dtype=np.int64,
    )

    real = (
        values.astype(np.float64)
        / float(1 << FRAC_BITS)
    )

    activated = (
        1.0
        / (1.0 + np.exp(-real))
    )

    return quantize_fixed(
        activated
    )
