"""Complex linear FFE benchmark, fitted by training-only least squares.

Uses the NN's identical X/Y rows, float32 input conversion and sequential split.
Tap order is current sample, previous sample, ... . No bias, nonlinear features,
reference-based test adaptation, or post-inference gain fitting is used.
"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from config import DATASET_PATH, TRAIN_RATIO

OUTPUT_PATH = Path(DATASET_PATH).parent / "equalized_output_ffe.mat"


def fit_predict_ffe(x, y, train_ratio=TRAIN_RATIO):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] % 2 or y.shape != (len(x), 2):
        raise ValueError("Expected interleaved I/Q X and two-column Y")
    split = int(train_ratio * len(x))
    if split < x.shape[1] // 2 or split >= len(x):
        raise ValueError("Insufficient training or test samples")
    # Match NN input precision, then solve robustly in complex double precision.
    taps = x[:, 0::2].astype(np.float64) + 1j*x[:, 1::2].astype(np.float64)
    reference = y[:, 0].astype(np.float64) + 1j*y[:, 1].astype(np.float64)
    weights, _, rank, singular_values = np.linalg.lstsq(
        taps[:split], reference[:split], rcond=None
    )
    prediction = taps[split:] @ weights
    return {
        "rx_equalized_ffe": prediction,
        "rx_before": taps[split:, 0],
        "tx_reference": reference[split:],
        "test_start": np.array([[split]], dtype=np.int32),
        "ffe_weights": weights,
        "ffe_taps": np.array([[taps.shape[1]]], dtype=np.int32),
        "training_rank": np.array([[rank]], dtype=np.int32),
        "training_singular_values": singular_values,
    }


def main():
    data = loadmat(DATASET_PATH)
    result = fit_predict_ffe(data["X"], data["Y"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    savemat(OUTPUT_PATH, result, do_compression=True)
    ref = result["tx_reference"]
    evm = 100*np.linalg.norm(result["rx_equalized_ffe"]-ref)/np.linalg.norm(ref)
    print(f"Complex LS FFE: {int(result['ffe_taps'][0, 0])} taps, held-out EVM {evm:.6f}%")
    print(f"MATLAB output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
