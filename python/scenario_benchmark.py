"""Independent FP32 research benchmark for MATLAB-generated PA scenarios.

Does not use or overwrite the existing ONNX/Q8.8 inference artifacts. All
receivers observe the same temporal window. Fits use training data, model
selection uses validation data, and the test set is evaluated only afterward.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat, savemat
from scipy.stats import beta
import torch
from torch import nn


def complex_values(iq):
    iq = np.asarray(iq)
    return iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)


def bit_error_interval(errors, count, confidence=0.95):
    """Two-sided Clopper-Pearson interval; zero errors remains an upper bound.

    This binomial interval is nominal: correlated errors may reduce effective
    sample size. Independent-burst replicate results are also retained.
    """
    if count <= 0 or not 0 <= errors <= count:
        raise ValueError("Expected 0 <= errors <= a positive bit count")
    tail = (1 - confidence) / 2
    low = 0.0 if errors == 0 else float(beta.ppf(tail, errors, count - errors + 1))
    high = 1.0 if errors == count else float(beta.ppf(1 - tail, errors + 1, count - errors))
    return low, high


def metrics(prediction, reference, ideal_symbols, ideal_bits):
    prediction = np.asarray(prediction).ravel()
    reference = np.asarray(reference).ravel()
    if prediction.shape != reference.shape or not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite and match the reference shape")
    energy = float(np.sum(np.abs(reference) ** 2))
    error_energy = float(np.sum(np.abs(prediction - reference) ** 2))
    detected = np.argmin(np.abs(prediction[:, None] - ideal_symbols[None, :]) ** 2, axis=1)
    expected = np.argmin(np.abs(reference[:, None] - ideal_symbols[None, :]) ** 2, axis=1)
    errors = int(np.count_nonzero(ideal_bits[detected] != ideal_bits[expected]))
    count = int(reference.size * ideal_bits.shape[1])
    low, high = bit_error_interval(errors, count)
    return dict(evm_percent=100 * math.sqrt(error_energy / energy),
                ber=errors / count, bit_errors=errors, bit_count=count,
                ber_lower_95=low, ber_upper_95=high,
                error_energy=error_energy, reference_energy=energy)


class PolynomialRegressor:
    """Full real I/Q Volterra polynomial, including all cross/lag terms.

    Features contain every unique monomial of degrees 1 through order. The
    intercept is unpenalized. Input and feature scaling use training data only.
    Ridge minimizes mean squared residual plus lambda * ||weights||^2.
    """

    def __init__(self, order):
        if order not in (1, 2, 3, 5):
            raise ValueError("Supported polynomial orders are 1, 2, 3, and 5")
        self.order = order

    def features(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.input_mean) / self.input_scale
        return np.column_stack([np.prod(z[:, indices], axis=1) for indices in self.terms])

    def fit(self, x, y, validation_x, validation_y, ridge_grid):
        x = np.asarray(x, dtype=np.float64)
        self.input_mean = x.mean(axis=0)
        self.input_scale = np.maximum(x.std(axis=0), 1e-12)
        self.terms = [indices for degree in range(1, self.order + 1)
                      for indices in itertools.combinations_with_replacement(range(x.shape[1]), degree)]
        phi = self.features(x)
        self.feature_mean = phi.mean(axis=0)
        self.feature_scale = np.maximum(phi.std(axis=0), 1e-12)
        phi = (phi - self.feature_mean) / self.feature_scale
        val = (self.features(validation_x) - self.feature_mean) / self.feature_scale
        self.target_mean = np.asarray(y, dtype=np.float64).mean(axis=0)
        target = y - self.target_mean
        u, singular, vt = np.linalg.svd(phi, full_matrices=False)
        projected = u.T @ target
        cutoff = np.finfo(float).eps * max(phi.shape) * singular[0]
        self.selection = []
        best = math.inf
        for ridge in ridge_grid:
            factors = np.zeros_like(singular)
            if ridge == 0:
                keep = singular > cutoff
                factors[keep] = 1 / singular[keep]
            else:
                factors = singular / (singular**2 + len(x) * ridge)
            weights = vt.T @ (factors[:, None] * projected)
            loss = float(np.mean((val @ weights + self.target_mean - validation_y) ** 2))
            self.selection.append(dict(ridge=float(ridge), validation_mse=loss))
            if loss < best:
                best = loss
                self.weights = weights
                self.ridge = float(ridge)
        self.rank = int(np.count_nonzero(singular > cutoff))
        return self

    def predict(self, x):
        phi = (self.features(x) - self.feature_mean) / self.feature_scale
        return phi @ self.weights + self.target_mean

    def save(self, path):
        powers = np.zeros((len(self.terms), len(self.input_mean)), dtype=np.int64)
        for row, term in enumerate(self.terms):
            for variable in term:
                powers[row, variable] += 1
        np.savez(path, input_mean=self.input_mean, input_scale=self.input_scale,
                 feature_mean=self.feature_mean, feature_scale=self.feature_scale,
                 target_mean=self.target_mean, weights=self.weights,
                 powers=powers, ridge=self.ridge, order=self.order)

    @property
    def parameter_count(self):
        return 2 * (len(self.terms) + 1)

    @property
    def multiplications(self):
        # Direct monomial construction plus the two output dot products.
        # Excludes affine normalization, which can be folded into coefficients.
        return sum(len(term) - 1 for term in self.terms) + 2 * len(self.terms)


def train_nn(x, y, validation_x, validation_y, cfg, seed):
    """Train the existing 4-12-2 topology with research-only Adam settings."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-12).astype(np.float32)
    inputs = torch.from_numpy(np.asarray((x - mean) / scale, dtype=np.float32))
    targets = torch.from_numpy(np.asarray(y, dtype=np.float32))
    val_x = torch.from_numpy(np.asarray((validation_x - mean) / scale, dtype=np.float32))
    val_y = torch.from_numpy(np.asarray(validation_y, dtype=np.float32))
    model = nn.Sequential(nn.Linear(x.shape[1], cfg['nn_hidden_size']),
                          nn.Sigmoid(), nn.Linear(cfg['nn_hidden_size'], 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['nn_learning_rate'])
    best_loss, best_epoch, history = math.inf, 0, []
    best_state = None
    for epoch in range(1, cfg['nn_epochs'] + 1):
        model.train()
        order = torch.randperm(len(inputs))
        loss_sum = 0.0
        for indices in order.split(cfg['nn_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(inputs[indices]), targets[indices])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(val_x), val_y).item()
        history.append((epoch, loss_sum / len(inputs), val_loss))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= cfg['nn_patience']:
            break
    if best_state is None:
        raise ValueError("NN training did not produce a finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, scale, np.asarray(history), best_epoch


def check_dataset(data):
    for partition in ('train', 'validation', 'test'):
        x, y = data['X_' + partition], data['Y_' + partition]
        if x.ndim != 2 or x.shape[1] != 4 or y.shape != (len(x), 2) or len(x) < 2:
            raise ValueError(f"Invalid {partition} shapes: X={x.shape}, Y={y.shape}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Nonfinite {partition} values")
    ideal = data['ideal_symbols'].ravel()
    bits = data['ideal_bits']
    if len(ideal) != 16 or bits.shape != (16, 4):
        raise ValueError("Scenario 1 requires MATLAB's 16-QAM constellation and bit labels")
    if not np.isin(bits, (0, 1)).all() or np.unique(bits, axis=0).shape[0] != 16:
        raise ValueError("Constellation labels must contain all 16 distinct four-bit words")
    for partition in ('train', 'validation', 'test'):
        target = complex_values(data['Y_' + partition])
        if np.max(np.min(abs(target[:, None] - ideal[None, :]), axis=1)) > 1e-6:
            raise ValueError(f"{partition} targets are not on the supplied constellation")


def run(dataset_path, output):
    torch.set_num_threads(1)
    data = loadmat(dataset_path)
    check_dataset(data)
    cfg = json.loads(str(data['configuration_json'].item()))
    seed = int(data['seed'].item())
    for key in ('nn_epochs', 'nn_patience', 'nn_batch_size', 'nn_hidden_size'):
        cfg[key] = int(cfg[key])
    output.mkdir(parents=True, exist_ok=True)
    x, y = data['X_train'], data['Y_train']
    vx, vy, tx = data['X_validation'], data['Y_validation'], data['X_test']
    ideal, bits = data['ideal_symbols'].ravel(), data['ideal_bits']
    reference = complex_values(data['Y_test'])
    predictions, rows, fits = [], [], {}

    def record(name, prediction, parameters, multiplications, activations=0):
        prediction = np.asarray(prediction)
        if prediction.ndim == 2:
            prediction = complex_values(prediction)
        predictions.append(prediction)
        row = dict(method=name, **metrics(prediction, reference, ideal, bits),
                   parameters=parameters, real_multiplications=multiplications,
                   sigmoid_evaluations=activations)
        rows.append(row)
        print(f"{name:18s} EVM={row['evm_percent']:8.4f}% "
              f"BER={row['ber']:.6g} ({row['bit_errors']}/{row['bit_count']})", flush=True)

    record('Before EQ', tx[:, :2], 0, 0)
    complex_x = x[:, 0::2].astype(float) + 1j*x[:, 1::2]
    complex_test = tx[:, 0::2].astype(float) + 1j*tx[:, 1::2]
    weights, _, rank, _ = np.linalg.lstsq(complex_x, complex_values(y), rcond=None)
    record('Complex FFE', complex_test @ weights, 2*len(weights), 4*len(weights))
    np.savez(output / 'complex_ffe.npz', weights=weights)
    fits['Complex FFE'] = dict(rank=int(rank))
    for order in (1, 2, 3, 5):
        name = 'Linear I/Q' if order == 1 else f'Volterra {order}'
        model = PolynomialRegressor(order).fit(x, y, vx, vy, cfg['ridge_grid'])
        record(name, model.predict(tx), model.parameter_count, model.multiplications)
        model.save(output / f'polynomial_order_{order}.npz')
        fits[name] = dict(ridge=model.ridge, rank=model.rank, selection=model.selection)

    model, mean, scale, history, best_epoch = train_nn(x, y, vx, vy, cfg, seed)
    with torch.no_grad():
        predicted = model(torch.from_numpy(np.asarray((tx - mean) / scale, dtype=np.float32))).numpy()
    hidden = cfg['nn_hidden_size']
    record('NN 4-12-2', predicted, sum(p.numel() for p in model.parameters()),
           hidden*(x.shape[1] + 2), hidden)
    torch.save(dict(state_dict=model.state_dict(), input_mean=torch.from_numpy(mean),
                    input_scale=torch.from_numpy(scale), input_size=x.shape[1],
                    hidden_size=hidden, activation='sigmoid', best_epoch=best_epoch),
               output / 'nn_fp32.pt')
    fits['NN 4-12-2'] = dict(best_epoch=best_epoch, epochs_completed=len(history),
                           optimizer='Adam', learning_rate=cfg['nn_learning_rate'],
                           best_validation_mse=float(history[best_epoch-1, 2]))
    np.savetxt(output / 'nn_history.csv', history, delimiter=',',
               header='epoch,training_mse,validation_mse', comments='')
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(schema_version=1, dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                    configuration=cfg, seed=seed, fits=fits,
                    partition_samples={key: len(data['X_' + key]) for key in ('train', 'validation', 'test')},
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__, torch=torch.__version__),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    arithmetic='FP32 neural network; FP64 linear/polynomial fits; no Q8.8 export',
                    complexity='Direct monomials and real MAC products; excludes preprocessing and sigmoid implementation',
                    ber_interval='Nominal 95% two-sided binomial Clopper-Pearson; errors may be correlated')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    savemat(output / 'results.mat', dict(method_names=np.asarray([r['method'] for r in rows], dtype=object),
                                        predictions=np.column_stack(predictions), reference=reference,
                                        history=history, best_epoch=best_epoch), do_compression=True)
    # results.mat is written last and serves as the condition completion marker.
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.output)
