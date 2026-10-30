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


class Q824LMS:
    """Two-output real/widely-linear LMS FFE, chronological batch-1 MSE.

    The complex-linear FFE is a subset of this mapping. All stored samples,
    weights, biases, deltas and learning rate are signed Q8.24 words.
    """
    def __init__(self, inputs, lr_q24):
        from hardware.training.fixedpoint import ONE
        self.inputs=inputs; self.lr_q24=int(lr_q24)
        self.weights=[[0]*inputs for _ in range(2)]; self.bias=[0,0]
        self.weights[0][0]=ONE; self.weights[1][1]=ONE

    def forward(self, x):
        from hardware.training import fixedpoint as fp
        return [fp.sat32(fp.round_shift((self.bias[j]<<fp.FRAC)+sum(w*v for w,v in zip(self.weights[j],x)),fp.FRAC),'lms_z') for j in range(2)]

    def step(self, x, target):
        from hardware.training import fixedpoint as fp
        output=self.forward(x)
        delta=[fp.sat32(a-b,'lms_delta') for a,b in zip(output,target)]
        for j,d in enumerate(delta):
            self.weights[j]=[fp.sat32(w-fp.qmul(fp.qmul(d,v,'lms_grad'),self.lr_q24),'lms_weight') for w,v in zip(self.weights[j],x)]
            self.bias[j]=fp.sat32(self.bias[j]-fp.qmul(d,self.lr_q24),'lms_bias')

    def predict(self, x):
        return np.asarray([self.forward(row) for row in x],dtype=np.int32)

    def state(self):
        return [row.copy() for row in self.weights],self.bias.copy()

    def restore(self, state):
        self.weights=[row.copy() for row in state[0]]; self.bias=state[1].copy()
