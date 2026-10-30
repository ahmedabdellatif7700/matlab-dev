"""Shared fixed-point primitives. Every value in the engine is a signed Q8.24 word (32 bit).

RTL counterpart: the single qmul / round-shift / saturate datapath reused by all atomics.
"""
from collections import Counter

FRAC = 24
ONE = 1 << FRAC
MAX32, MIN32 = (1 << 31) - 1, -(1 << 31)
stats = Counter()          # saturation events by tag (checks only)


def round_shift(value, shift):
    """Divide by 2**shift, rounding to nearest, ties away from zero."""
    if shift == 0:
        return value
    half = 1 << (shift - 1)
    return (value + half) >> shift if value >= 0 else -((-value + half) >> shift)


def sat32(value, tag=""):
    if value > MAX32:
        stats[tag] += 1
        return MAX32
    if value < MIN32:
        stats[tag] += 1
        return MIN32
    return value


def qmul(a, b, tag=""):
    """Q8.24 x Q8.24 -> Q8.24: 64-bit product, one rounding, saturate."""
    return sat32(round_shift(a * b, FRAC), tag)


def to_q(value):
    return sat32(int(round(value * ONE)))
