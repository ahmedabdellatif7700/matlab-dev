#!/usr/bin/env python3
"""Verify software SGD training snapshots and their RTL memory conventions.

This utility verifies training artifacts; the SGD implementation is separate.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import equalizer as eq

LAYOUT = (
    ("fc1_weights", 48),
    ("fc1_biases", 12),
    ("fc2_weights", 24),
    ("fc2_biases", 2),
)
FP32_FILES = {
    "train_inputs.mem", "train_targets.mem", "parameters_before.mem",
    "gradients.mem", "parameters_after.mem",
}


def read_hex(path, count, width):
    lines = path.read_text().splitlines()
    if len(lines) != count:
        raise ValueError(f"{path}: expected {count} words, got {len(lines)}")
    if any(len(word) != width or any(c not in "0123456789ABCDEF" for c in word)
           for word in lines):
        raise ValueError(f"{path}: expected {width} uppercase hexadecimal digits/word")
    return np.asarray([int(word, 16) for word in lines],
                      dtype=np.uint16 if width == 4 else np.uint32)


def fp32(root, name, count):
    return read_hex(root / name, count, 8).view(np.float32)


def check_rtl(root, manifest, values):
    expected = {
        f"rtl/{stage}/{label}.mem": count
        for stage in ("before", "after")
        for label, count in LAYOUT
    }
    expected.update({
        "rtl/before/parameters.mem": 128,
        "rtl/after/parameters.mem": 128,
        "rtl/train_inputs.mem": manifest["samples_in_batch"] * 4,
        "rtl/train_targets.mem": manifest["samples_in_batch"] * 2,
    })
    if manifest["rtl_file_counts"] != expected:
        raise ValueError("Incorrect RTL memory names or counts")

    for stage in ("before", "after"):
        packed = []
        offset = 0
        for label, count in LAYOUT:
            rel = f"rtl/{stage}/{label}.mem"
            actual = read_hex(root / rel, count, 4)
            expected_values = eq.quantize(
                values[f"parameters_{stage}.mem"][offset:offset + count]
            ).astype(np.int16).view(np.uint16)
            if not np.array_equal(actual, expected_values):
                raise ValueError(f"{rel}: differs from original equalizer quantizer")
            packed.extend(actual.tolist())
            offset += count

        image = read_hex(root / f"rtl/{stage}/parameters.mem", 128, 4)
        if not np.array_equal(image[:86], np.asarray(packed, dtype=np.uint16)):
            raise ValueError(f"rtl/{stage}/parameters.mem: parameter order mismatch")
        if not np.array_equal(image[86:], np.zeros(42, dtype=np.uint16)):
            raise ValueError(f"rtl/{stage}/parameters.mem: 42 zero words missing")

    for name in ("train_inputs.mem", "train_targets.mem"):
        rel = f"rtl/{name}"
        actual = read_hex(root / rel, expected[rel], 4)
        expected_values = eq.quantize_fixed(values[name]).astype(np.int16).view(np.uint16)
        if not np.array_equal(actual, expected_values):
            raise ValueError(f"{rel}: differs from original equalizer quantizer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "artifacts" / "training" / "fp32_sgd")
    args = parser.parse_args()
    root = args.output
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["optimizer"] != "SGD" or manifest["epochs_configured"] != 1:
        raise ValueError("Expected one-epoch SGD reference")
    if manifest["momentum"] != 0 or manifest["weight_decay"] != 0:
        raise ValueError("Expected plain SGD without momentum or weight decay")
    if manifest["shuffling"] is not False:
        raise ValueError("Expected deterministic sequential sample ordering")

    count = manifest["samples_in_batch"]
    expected_counts = {
        "train_inputs.mem": count * 4,
        "train_targets.mem": count * 2,
        "parameters_before.mem": 86,
        "gradients.mem": 86,
        "parameters_after.mem": 86,
    }
    if manifest["file_counts"] != expected_counts:
        raise ValueError("Incorrect FP32 memory names or counts")

    values = {name: fp32(root, name, size) for name, size in expected_counts.items()}
    for name, data in values.items():
        if not np.isfinite(data).all():
            raise ValueError(f"{name}: contains NaN/Inf")

    before = values["parameters_before.mem"]
    grad = values["gradients.mem"]
    after = values["parameters_after.mem"]
    lr = np.float32(manifest["learning_rate"])
    expected_after = before - lr * grad
    np.testing.assert_allclose(
        after, expected_after, rtol=2e-6, atol=1e-7,
        err_msg="FP32 parameter update does not match plain SGD",
    )
    if not (0 < manifest["captured_step"] <= math.ceil(
            manifest["selected_samples"] / manifest["batch_size"])):
        raise ValueError("Captured step outside selected epoch")

    check_rtl(root, manifest, values)

    expected_paths = (FP32_FILES | {"manifest.json"} |
                      set(manifest["rtl_file_counts"]))
    actual_paths = {str(path.relative_to(root))
                    for path in root.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise ValueError(f"Unexpected/missing generated files: "
                         f"{actual_paths ^ expected_paths}")

    print("[PASS] FP32 SGD update: after = before - learning_rate * gradient")
    print("[PASS] All 12 Q8.8 RTL .mem files match original quantization/layout")
    print("[PASS] 86 active parameters + 42 reserved zeros in 128-word images")
    print(f"[PASS] Captured SGD update {manifest['captured_step']}; "
          f"epoch_completed={manifest['epoch_completed']}")


if __name__ == "__main__":
    main()
