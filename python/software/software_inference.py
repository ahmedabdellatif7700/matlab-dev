"""Regenerate Q8.8 software inference outputs using saved experiment artifacts.

Only standard sigmoid currently provides bit-exact KV260 RTL predictions.
Alpha-based activation variants use the trained alpha saved in the NPZ model.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import sys

# Permit direct execution: python software/software_inference.py
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))
from config import INFERENCE_DIR, QUANT_PATH, ACTIVATION
from hardware.q8_8_inference_arithmetic import (
    mac_q8_8, sigmoid_q8_8, quantize_fixed, dequantize_fixed,
)
from activations.variable_alpha_sigmoid import variable_alpha_sigmoid
from activations.stable_alpha_sigmoid import stable_alpha_sigmoid


COLUMNS = ("sample", "output_0_q8", "output_1_q8", "output_0", "output_1")


def read_q8_mem(path: Path, expected_words: int | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Q8.8 memory file: {path}")
    tokens = path.read_text(encoding="ascii").split()
    if expected_words is not None and len(tokens) != expected_words:
        raise ValueError(f"{path}: expected {expected_words} words, found {len(tokens)}")
    try:
        unsigned = np.asarray([int(token, 16) for token in tokens], dtype=np.int64)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid hexadecimal word") from exc
    if np.any((unsigned < 0) | (unsigned > 0xFFFF)):
        raise ValueError(f"{path}: word exceeds 16 bits")
    return np.where(unsigned >= 0x8000, unsigned - 0x10000, unsigned)


def load_parameters(combined_path: Path) -> np.ndarray:
    """Read the 128-word packed FPGA parameter image (86 active + 42 reserved)."""
    words = read_q8_mem(combined_path, expected_words=128)
    if np.any(words[86:] != 0):
        raise ValueError(f"{combined_path}: expected 42 reserved zero words")
    print(f"Parameters: combined {combined_path}")
    return words[:86]


def predict(inputs: np.ndarray, parameters: np.ndarray, *, activation: str = "sigmoid", alpha: float | None = None) -> np.ndarray:
    if inputs.ndim != 2 or inputs.shape[1] != 4 or parameters.shape != (86,):
        raise ValueError("Expected (N, 4) inputs and 86 Q8.8 parameters")
    hidden = mac_q8_8(inputs, parameters[:48].reshape(12, 4), parameters[48:60])
    if activation == "sigmoid":
        hidden = sigmoid_q8_8(hidden)
    elif activation in ("variable_alpha", "stable_alpha"):
        if alpha is None:
            raise ValueError(f"{activation} requires the saved Q8.8 alpha coefficient")
        hidden_float = dequantize_fixed(hidden)
        if activation == "variable_alpha":
            hidden_float = variable_alpha_sigmoid(hidden_float, alpha=alpha)
        else:
            hidden_float = stable_alpha_sigmoid(hidden_float, alpha=alpha)
        hidden = quantize_fixed(hidden_float)
    else:
        raise ValueError(f"Unsupported software activation: {activation}")
    outputs = mac_q8_8(hidden, parameters[60:84].reshape(2, 12), parameters[84:86])
    return np.asarray(outputs, dtype=np.int64)


def write_csv(path: Path, outputs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for index, (q0, q1) in enumerate(outputs):
            writer.writerow((index, int(q0), int(q1), f"{q0 / 256.0:.8f}", f"{q1 / 256.0:.8f}"))


def compare_hardware(path: Path, software: np.ndarray) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in COLUMNS):
            raise ValueError(f"{path}: missing required CSV columns")
        rows = list(reader)

    if len(rows) != len(software):
        raise ValueError(f"Sample count mismatch: hardware={len(rows)}, software={len(software)}")
    hardware = np.empty_like(software)
    for index, row in enumerate(rows):
        if int(row["sample"]) != index:
            raise ValueError(f"Hardware CSV sample index mismatch on row {index}")
        hardware[index, 0] = int(row["output_0_q8"])
        hardware[index, 1] = int(row["output_1_q8"])

    differences = software != hardware
    mismatch_count = int(np.count_nonzero(differences))
    print(f"Compared {len(software)} samples / {software.size} Q8.8 outputs")
    print(f"Q8.8 mismatches: {mismatch_count}")
    if mismatch_count:
        for row, column in np.argwhere(differences)[:10]:
            print(f"  sample {row}, output {column}: software={software[row, column]}, "
                  f"hardware={hardware[row, column]}")
        return 1
    print("[PASS] Bit-exact software vs KV260 hardware inference")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INFERENCE_DIR / "test_inputs.mem")
    parser.add_argument("--parameters", type=Path, default=INFERENCE_DIR / "parameters.mem")
    parser.add_argument("--output", type=Path, default=INFERENCE_DIR / "software_outputs.csv")
    parser.add_argument("--hardware", type=Path, help="Optional KV260 hardware_outputs.csv for exact comparison")
    args = parser.parse_args()

    words = read_q8_mem(args.inputs)
    if words.size == 0 or words.size % 4 != 0:
        raise ValueError(f"{args.inputs}: input count must be a nonzero multiple of four")
    inputs = words.reshape(-1, 4)
    parameters = load_parameters(args.parameters)
    alpha = None
    if ACTIVATION != "sigmoid":
        if not QUANT_PATH.is_file():
            raise FileNotFoundError(f"Missing alpha and model data: {QUANT_PATH}")
        with np.load(QUANT_PATH) as model:
            if "alpha" not in model:
                raise ValueError(f"No quantized alpha saved in {QUANT_PATH}")
            alpha = float(np.asarray(model["alpha"]).item()) / 256.0
        print(f"Activation: {ACTIVATION}, saved Q8.8 alpha={alpha:.8f}")
        print("NOTE: Not a bit-exact reference for the sigmoid-only FPGA RTL")
    outputs = predict(inputs, parameters, activation=ACTIVATION, alpha=alpha)
    write_csv(args.output, outputs)
    print(f"Software inference: {len(outputs)} samples, {outputs.size} Q8.8 outputs")
    print(f"Saved: {args.output}")
    if args.hardware and ACTIVATION != "sigmoid":
        raise ValueError("FPGA comparison requires the supported standard-sigmoid activation")
    return compare_hardware(args.hardware, outputs) if args.hardware else 0


if __name__ == "__main__":
    raise SystemExit(main())
