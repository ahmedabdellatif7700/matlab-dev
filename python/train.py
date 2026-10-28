#!/usr/bin/env python3
"""Train the unified Q8.24 model on the real dataset and compare it with float64 SGD.

Usage: python train.py --dataset training_data.mat [--epochs 10] [--seed 42]
Float reference uses lr = 2**-LR_SHIFT, same init, same sample order, batch size 1.
"""
import argparse, time
from pathlib import Path
import numpy as np
import scipy.io
from hardware.training import fixedpoint as fp
from hardware.training import memory_map as mm
from hardware.training.controller import run_phase, train_step
from hardware.training.update_atomic import LR_SHIFT


def init_params(seed):
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-0.5, 0.5, 48); b1 = rng.uniform(-0.5, 0.5, 12)
    s = 1 / np.sqrt(12)
    w2 = rng.uniform(-s, s, 24); b2 = rng.uniform(-s, s, 2)
    return w1, b1, w2, b2


def load_ram(ram, w1, b1, w2, b2):
    for base, v in ((mm.W0, w1), (mm.B0, b1), (mm.W1, w2), (mm.B1, b2)):
        for i, e in enumerate(v):
            ram[base + i] = fp.to_q(float(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", type=Path, default=None,
                    help="init_params.txt from dump_init.py (86 values); default: numpy init")
    ap.add_argument("--lr-shift", type=int, default=LR_SHIFT)
    a = ap.parse_args()
    lr = 2.0 ** -a.lr_shift

    d = scipy.io.loadmat(a.dataset)
    X, Y = d["X"].astype(np.float64), d["Y"].astype(np.float64)
    n = int(0.8 * len(X))
    Xtr, Ytr, Xte, Yte = X[:n], Y[:n], X[n:], Y[n:]
    print(f"data |x|max={abs(X).max():.2f} |t|max={abs(Y).max():.2f} (Q8.24 range 128)")
    qx = lambda r: [fp.to_q(float(e)) for e in r]

    if a.init:
        v = np.loadtxt(a.init)
        assert v.size == 86, "init file must hold 86 values"
        w1, b1, w2, b2 = v[:48], v[48:60], v[60:84], v[84:86]
    else:
        w1, b1, w2, b2 = init_params(a.seed)
    ram = mm.new_ram(); load_ram(ram, w1, b1, w2, b2)
    # float reference starts from the same quantized values
    P = np.array(ram[:86], dtype=np.float64) / fp.ONE
    W1, B1 = P[:48].reshape(12, 4).copy(), P[48:60].copy()
    W2, B2 = P[60:84].reshape(2, 12).copy(), P[84:86].copy()
    sig = lambda z: 1 / (1 + np.exp(-z))

    def int_eval():
        sc = list(ram); ys = []
        for x in Xte:
            sc[mm.A0:mm.A0 + 4] = qx(x); run_phase(sc, "fwd"); ys.append(sc[mm.A2:mm.A2 + 2])
        ys = np.array(ys, dtype=np.float64) / fp.ONE
        return np.mean((ys - Yte) ** 2), np.mean(np.sign(ys) != np.sign(Yte))

    def float_eval():
        y = sig(Xte @ W1.T + B1) @ W2.T + B2
        return np.mean((y - Yte) ** 2), np.mean(np.sign(y) != np.sign(Yte))

    Xq = [qx(r) for r in Xtr]; Yq = [qx(r) for r in Ytr]
    print(f"train={n} test={len(Xte)} lr=2^-{a.lr_shift}={lr:.6f}")
    m, b = int_eval(); mf, bf = float_eval()
    print(f"epoch  0: int MSE {m:.4f} BER {b:.4f} | float MSE {mf:.4f} BER {bf:.4f}")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        for x, t in zip(Xq, Yq):
            ram[mm.A0:mm.A0 + 4] = x; ram[mm.T:mm.T + 2] = t
            train_step(ram, a.lr_shift)
        for x, t in zip(Xtr, Ytr):
            h = sig(W1 @ x + B1); e = W2 @ h + B2 - t
            dh = (W2.T @ e) * h * (1 - h)
            W2 -= lr * np.outer(e, h); B2 -= lr * e
            W1 -= lr * np.outer(dh, x); B1 -= lr * dh
        m, b = int_eval(); mf, bf = float_eval()
        print(f"epoch {ep:2d}: int MSE {m:.4f} BER {b:.4f} | float MSE {mf:.4f} BER {bf:.4f}"
              f" | sat {dict(fp.stats)} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
