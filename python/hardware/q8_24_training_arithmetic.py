#!/usr/bin/env python3
"""Hardware-equivalent, integer-only SGD for the KV260 4-12-2 equalizer.

Numerical contract: NUMERICAL_SPEC.md, draft v0.1 (2026-10-15).
This is a *proposed* surrogate-gradient fixed-point training model; no training
RTL is implemented or verified yet. It preserves the Q8.8 forward behavior of
the existing inference accelerator and accumulates SGD changes in Q8.24.

Pure model operations use Python integers (no float arithmetic). Float-to-Q8.8
conversion is allowed only at the external dataset boundary. Run this module
as a script for baseline inference checks or a small deterministic SGD run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

INPUTS = 4
HIDDEN = 12
OUTPUTS = 2
ACTIVE_PARAMETERS = 86
MEMORY_DEPTH = 128
Q8_FRAC = 8
MASTER_FRAC = 24
GRAD_FRAC = 16
LR_Q24 = 16777  # round(0.001 * 2**24), stored as unsigned Q0.24

# Extracted in order from rtl/core/sigmoid.v (2026-10-15).
# Index i holds the *inclusive* maximum input yielding output i; values
# greater than the last threshold yield 256. Never approximate using exp().
SIGMOID_THRESHOLDS_Q8 = (
    -1597, -1315, -1183, -1096, -1030,  -978,  -934,  -897,
     -864,  -834,  -807,  -783,  -761,  -740,  -721,  -702,
     -685,  -669,  -654,  -639,  -625,  -612,  -599,  -587,
     -575,  -564,  -553,  -543,  -532,  -522,  -513,  -503,
     -494,  -485,  -477,  -468,  -460,  -452,  -444,  -436,
     -428,  -421,  -414,  -407,  -400,  -393,  -386,  -379,
     -373,  -366,  -360,  -354,  -347,  -341,  -335,  -329,
     -323,  -318,  -312,  -306,  -301,  -295,  -290,  -284,
     -279,  -274,  -269,  -263,  -258,  -253,  -248,  -243,
     -238,  -233,  -228,  -224,  -219,  -214,  -209,  -205,
     -200,  -195,  -191,  -186,  -182,  -177,  -173,  -168,
     -164,  -159,  -155,  -151,  -146,  -142,  -138,  -133,
     -129,  -125,  -121,  -116,  -112,  -108,  -104,  -100,
      -96,   -91,   -87,   -83,   -79,   -75,   -71,   -67,
      -63,   -59,   -55,   -51,   -47,   -43,   -39,   -35,
      -31,   -27,   -23,   -19,   -15,   -11,    -7,    -3,
        2,     6,    10,    14,    18,    22,    26,    30,
       34,    38,    42,    46,    50,    54,    58,    62,
       66,    70,    74,    78,    82,    86,    90,    95,
       99,   103,   107,   111,   115,   120,   124,   128,
      132,   137,   141,   145,   150,   154,   158,   163,
      167,   172,   176,   181,   185,   190,   194,   199,
      204,   208,   213,   218,   223,   227,   232,   237,
      242,   247,   252,   257,   262,   268,   273,   278,
      283,   289,   294,   300,   305,   311,   317,   322,
      328,   334,   340,   346,   353,   359,   365,   372,
      378,   385,   392,   399,   406,   413,   420,   427,
      435,   443,   451,   459,   467,   476,   484,   493,
      502,   512,   521,   531,   542,   552,   563,   574,
      586,   598,   611,   624,   638,   653,   668,   684,
      701,   720,   739,   760,   782,   806,   833,   863,
      896,   933,   977,  1029,  1095,  1182,  1314,  1596,
)


def require_signed(value: int, bits: int, label: str) -> int:
    """Reject values outside a signed hardware input/word width."""
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer, got {type(value).__name__}")
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if not minimum <= value <= maximum:
        raise ValueError(f"{label}={value} outside signed {bits}-bit range")
    return value


def sat_signed(value: int, bits: int) -> int:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    return min(max(value, minimum), maximum)


def round_shift(value: int, shift: int) -> int:
    """Round signed integer right shift to nearest, ties away from zero."""
    if shift < 0:
        raise ValueError("shift must be nonnegative")
    if shift == 0:
        return value
    rounded_magnitude = (abs(value) + (1 << (shift - 1))) >> shift
    return -rounded_magnitude if value < 0 else rounded_magnitude


def quantize_external_q8(value: float) -> int:
    """Only for float import: nearest ties-to-even, as np.rint in fixed_point.py."""
    return sat_signed(round(float(value) * (1 << Q8_FRAC)), 16)


def signed_word(unsigned_value: int, bits: int) -> int:
    """Interpret a fixed-width unsigned memory word as two's complement."""
    if not 0 <= unsigned_value < (1 << bits):
        raise ValueError("memory word exceeds declared width")
    return unsigned_value - (1 << bits) if unsigned_value & (1 << (bits - 1)) else unsigned_value


def read_mem(path: str | Path, bits: int, expected_words: int | None = None) -> list[int]:
    """Read one two's-complement hex word per line, preserving signed values."""
    if bits % 4:
        raise ValueError("memory word width must be a multiple of four")
    path = Path(path)
    words: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        word = line.strip()
        if not word:
            continue
        if len(word) != bits // 4 or any(c not in "0123456789abcdefABCDEF" for c in word):
            raise ValueError(f"{path}:{line_number}: expected {bits // 4} hexadecimal digits")
        words.append(signed_word(int(word, 16), bits))
    if expected_words is not None and len(words) != expected_words:
        raise ValueError(f"{path}: expected {expected_words} words, got {len(words)}")
    return words


def write_mem(path: str | Path, words: Sequence[int], bits: int) -> None:
    """Export two's-complement uppercase hex without altering source arrays."""
    mask = (1 << bits) - 1
    lines = []
    for index, word in enumerate(words):
        require_signed(word, bits, f"word[{index}]")
        lines.append(f"{word & mask:0{bits // 4}X}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def sigmoid_q8(z_q8: int) -> int:
    """Exactly reproduce rtl/core/sigmoid.v threshold-chain output."""
    require_signed(z_q8, 16, "sigmoid input")
    # bisect_left selects the first inclusive threshold >= input.
    from bisect import bisect_left
    return bisect_left(SIGMOID_THRESHOLDS_Q8, z_q8)


def mac_q8(inputs_q8: Sequence[int], weights_q8: Sequence[int], bias_q8: int) -> int:
    """Existing 40-bit MAC: signed Q8.8 products, one rounding, 16-bit saturation."""
    if len(inputs_q8) != len(weights_q8):
        raise ValueError("MAC input and weight counts differ")
    if len(inputs_q8) not in (INPUTS, HIDDEN):
        raise ValueError("MAC fan-in must be 4 or 12")
    require_signed(bias_q8, 16, "MAC bias")
    acc_q16 = bias_q8 << Q8_FRAC
    for i, (x, w) in enumerate(zip(inputs_q8, weights_q8)):
        require_signed(x, 16, f"MAC input[{i}]")
        require_signed(w, 16, f"MAC weight[{i}]")
        acc_q16 += x * w
    require_signed(acc_q16, 40, "MAC accumulator")
    return sat_signed(round_shift(acc_q16, Q8_FRAC), 16)


@dataclass(frozen=True)
class ForwardTrace:
    hidden_pre_q8: tuple[int, ...]       # 12, before sigmoid
    hidden_q8: tuple[int, ...]           # 12, after sigmoid
    predictions_q8: tuple[int, ...]      # 2, linear outputs


@dataclass(frozen=True)
class StepTrace:
    inputs_q8: tuple[int, ...]
    targets_q8: tuple[int, ...]
    master_before_q24: tuple[int, ...]   # 86, pre-update
    published_before_q8: tuple[int, ...] # 128, pre-update
    forward: ForwardTrace
    errors_q8: tuple[int, ...]           # 2, 32-bit with 8 fractional bits
    loss_q16: int                        # 64-bit container, 16 fractional bits
    output_deltas_q16: tuple[int, ...]   # 2
    sigmoid_derivatives_q16: tuple[int, ...] # 12
    backprop_sums_q24: tuple[int, ...]   # 12, 64-bit intermediate
    hidden_deltas_q16: tuple[int, ...]   # 12
    gradients_q16: tuple[int, ...]       # 86, address ordered
    master_after_q24: tuple[int, ...]    # 86
    published_after_q8: tuple[int, ...]  # 128
    changed_master_words: int
    changed_published_words: int
    gradient_saturations: int
    delta_saturations: int
    master_saturations: int


def forward_q8(inputs_q8: Sequence[int], published_q8: Sequence[int]) -> ForwardTrace:
    """Run the unchanged 4-12-2 Q8.8 inference computation."""
    if len(inputs_q8) != INPUTS or len(published_q8) != MEMORY_DEPTH:
        raise ValueError("forward requires four inputs and a 128-word parameter image")
    for index, word in enumerate(published_q8):
        require_signed(word, 16, f"published[{index}]")
    hidden_pre = tuple(
        mac_q8(inputs_q8, published_q8[j * INPUTS:(j + 1) * INPUTS], published_q8[48 + j])
        for j in range(HIDDEN)
    )
    hidden = tuple(sigmoid_q8(z) for z in hidden_pre)
    predictions = tuple(
        mac_q8(hidden, published_q8[60 + k * HIDDEN:60 + (k + 1) * HIDDEN], published_q8[84 + k])
        for k in range(OUTPUTS)
    )
    return ForwardTrace(hidden_pre, hidden, predictions)


class FixedPointSGD:
    """Online surrogate SGD with Q8.24 masters and coherent Q8.8 publication.

    The master and published images are independent arrays. A training step
    computes all gradients from the same pre-update Q8.8 snapshot, then
    commits all master changes and publishes all Q8.8 words together.
    """

    def __init__(self, initial_q8: Sequence[int]):
        if len(initial_q8) != MEMORY_DEPTH:
            raise ValueError(f"expected {MEMORY_DEPTH} inference parameter words")
        self.published_q8 = [require_signed(v, 16, f"parameter[{i}]") for i, v in enumerate(initial_q8)]
        if any(self.published_q8[ACTIVE_PARAMETERS:]):
            raise ValueError("reserved parameter words 86..127 must be zero")
        self.master_q24 = [value << (MASTER_FRAC - Q8_FRAC) for value in self.published_q8[:ACTIVE_PARAMETERS]]
        self.steps = 0

    @classmethod
    def from_mem(cls, path: str | Path) -> "FixedPointSGD":
        return cls(read_mem(path, 16, MEMORY_DEPTH))

    def forward(self, inputs_q8: Sequence[int]) -> ForwardTrace:
        return forward_q8(inputs_q8, self.published_q8)

    def step(self, inputs_q8: Sequence[int], targets_q8: Sequence[int]) -> StepTrace:
        """Execute one deterministic, fully integer SGD step (batch size 1)."""
        if len(inputs_q8) != INPUTS or len(targets_q8) != OUTPUTS:
            raise ValueError("step expects four Q8.8 inputs and two Q8.8 targets")
        x = tuple(require_signed(v, 16, f"input[{i}]") for i, v in enumerate(inputs_q8))
        t = tuple(require_signed(v, 16, f"target[{i}]") for i, v in enumerate(targets_q8))
        before_q8 = tuple(self.published_q8)
        before_master = tuple(self.master_q24)
        forward = forward_q8(x, before_q8)

        # Two-output MSE: loss = (e0**2 + e1**2)/2.
        # For this definition, dLoss/dPrediction[k] = e[k].
        errors = tuple(forward.predictions_q8[k] - t[k] for k in range(OUTPUTS))
        for k, error in enumerate(errors):
            require_signed(error, 32, f"error[{k}]")
        loss_q16 = round_shift(sum(error * error for error in errors), 1)
        require_signed(loss_q16, 64, "loss")
        delta2 = tuple(error << (GRAD_FRAC - Q8_FRAC) for error in errors)
        for k, delta in enumerate(delta2):
            require_signed(delta, 32, f"output delta[{k}]")

        # Straight-through/surrogate derivative: h*(1-h), not the
        # mathematical derivative of the discrete sigmoid threshold table.
        derivative = tuple(h * (256 - h) for h in forward.hidden_q8)
        backprop_sums = tuple(
            before_q8[60 + j] * delta2[0] + before_q8[72 + j] * delta2[1]
            for j in range(HIDDEN)
        )
        delta_saturations = 0
        hidden_deltas = []
        for j in range(HIDDEN):
            require_signed(backprop_sums[j], 64, f"backprop sum[{j}]")
            back_raw = round_shift(backprop_sums[j], 8)
            back = sat_signed(back_raw, 32)
            delta_saturations += (back != back_raw)
            hidden_raw = round_shift(back * derivative[j], 16)
            hidden_delta = sat_signed(hidden_raw, 32)
            delta_saturations += (hidden_delta != hidden_raw)
            hidden_deltas.append(hidden_delta)
        delta1 = tuple(hidden_deltas)

        # Parameter gradients in exact memory order: W1, b1, W2, b2.
        raw_gradients = (
            [round_shift(delta1[j] * x[i], 8) for j in range(HIDDEN) for i in range(INPUTS)]
            + list(delta1)
            + [round_shift(delta2[k] * forward.hidden_q8[j], 8)
               for k in range(OUTPUTS) for j in range(HIDDEN)]
            + list(delta2)
        )
        if len(raw_gradients) != ACTIVE_PARAMETERS:
            raise AssertionError("internal gradient ordering/count error")
        gradients = tuple(sat_signed(g, 32) for g in raw_gradients)
        gradient_saturations = sum(g != raw for g, raw in zip(gradients, raw_gradients))

        # Updates use the same pre-update master image, with a 64-bit
        # multiplication and Q16.40 -> Q8.24 rounding shift.
        after_master = []
        master_saturations = 0
        for old, grad in zip(before_master, gradients):
            product_q40 = grad * LR_Q24
            require_signed(product_q40, 64, "SGD product")
            update_q24 = round_shift(product_q40, 16)
            next_raw = old - update_q24
            next_master = sat_signed(next_raw, 32)
            master_saturations += (next_raw != next_master)
            after_master.append(next_master)

        # Publish the *complete* new 86-word image before the next forward.
        after_q8 = [sat_signed(round_shift(v, MASTER_FRAC - Q8_FRAC), 16)
                    for v in after_master]
        after_q8.extend([0] * (MEMORY_DEPTH - ACTIVE_PARAMETERS))
        trace = StepTrace(
            inputs_q8=x,
            targets_q8=t,
            master_before_q24=before_master,
            published_before_q8=before_q8,
            forward=forward,
            errors_q8=errors,
            loss_q16=loss_q16,
            output_deltas_q16=delta2,
            sigmoid_derivatives_q16=derivative,
            backprop_sums_q24=backprop_sums,
            hidden_deltas_q16=delta1,
            gradients_q16=gradients,
            master_after_q24=tuple(after_master),
            published_after_q8=tuple(after_q8),
            changed_master_words=sum(a != b for a, b in zip(before_master, after_master)),
            changed_published_words=sum(a != b for a, b in zip(before_q8, after_q8)),
            gradient_saturations=gradient_saturations,
            delta_saturations=delta_saturations,
            master_saturations=master_saturations,
        )
        self.master_q24 = after_master
        self.published_q8 = after_q8
        self.steps += 1
        return trace


def verify_inference(
    model: FixedPointSGD, input_mem: str | Path, expected_mem: str | Path
) -> int:
    """Compare the current published weights with existing inference vectors."""
    inputs = read_mem(input_mem, 16)
    expected = read_mem(expected_mem, 16)
    if len(inputs) % INPUTS or len(expected) != (len(inputs) // INPUTS) * OUTPUTS:
        raise ValueError("input/output vector counts are inconsistent")
    for sample in range(len(inputs) // INPUTS):
        x = inputs[sample * INPUTS:(sample + 1) * INPUTS]
        y = model.forward(x).predictions_q8
        expected_y = expected[sample * OUTPUTS:(sample + 1) * OUTPUTS]
        if list(y) != expected_y:
            raise AssertionError(f"sample {sample}: actual={y}, expected={expected_y}")
    return len(inputs) // INPUTS


def load_training_mat(path: str | Path, start: int, count: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Import a deterministic training prefix; SciPy is needed only here."""
    import scipy.io  # Optional dependency: not used by the integer training core.

    data = scipy.io.loadmat(path)
    x, y = data["X"], data["Y"]
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != INPUTS or y.shape[1] != OUTPUTS:
        raise ValueError("expected dataset X=(N,4), Y=(N,2)")
    if len(x) != len(y):
        raise ValueError("dataset input/target sample counts differ")
    # Same sequential 80/20 split as equalizer.py. No shuffle.
    train_size = int(0.8 * len(x))
    if start < 0 or count < 1 or start + count > train_size:
        raise ValueError(f"requested samples outside training subset [0, {train_size})")
    samples = []
    for index in range(start, start + count):
        input_words = tuple(quantize_external_q8(v) for v in x[index])
        target_words = tuple(quantize_external_q8(v) for v in y[index])
        samples.append((input_words, target_words))
    return samples


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=root / "python/artifacts/inference/parameters.mem")
    parser.add_argument("--dataset", type=Path, default=root / "datasets/training_data.mat")
    parser.add_argument("--steps", type=int, default=1, help="number of online SGD updates")
    parser.add_argument("--start-index", type=int, default=0, help="first training sample index")
    parser.add_argument("--verify-inference", action="store_true",
                        help="check original inference vectors instead of training")
    parser.add_argument("--test-inputs", type=Path, default=root / "python/artifacts/inference/test_inputs.mem")
    parser.add_argument("--expected-outputs", type=Path, default=root / "memory/expected_outputs.mem")
    args = parser.parse_args()

    model = FixedPointSGD.from_mem(args.parameters)
    if args.verify_inference:
        count = verify_inference(model, args.test_inputs, args.expected_outputs)
        print(f"PASS: {count} baseline samples, {count * OUTPUTS} bit-exact Q8.8 outputs")
        return

    samples = load_training_mat(args.dataset, args.start_index, args.steps)
    for index, (x, t) in enumerate(samples, start=args.start_index):
        trace = model.step(x, t)
        print(
            f"sample={index} loss_q16={trace.loss_q16} "
            f"loss={trace.loss_q16 / 65536:.8f} "
            f"master_changed={trace.changed_master_words} "
            f"q8_changed={trace.changed_published_words} "
            f"grad_sat={trace.gradient_saturations} "
            f"delta_sat={trace.delta_saturations} "
            f"master_sat={trace.master_saturations}"
        )
    print(f"Completed {model.steps} integer-only SGD update(s); original files unchanged")


if __name__ == "__main__":
    main()
