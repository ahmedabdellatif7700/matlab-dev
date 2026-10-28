"""Held-out research receivers shared by the MZM/optical/multiuser/OFDM studies.

MATLAB owns waveform generation, QAM labels, front-end processing and splits.
No scenario reads or overwrites the legacy hardware inference artifacts.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat, savemat
import torch
from torch import nn

from scenario_benchmark import PolynomialRegressor, metrics
from activations.activation import MultilevelSigmoid


class StableMultilevel(MultilevelSigmoid):
    """The same piecewise function, evaluated without overflowing exp()."""
    def forward(self, x):
        level = torch.ceil((x-1)/2).clamp(-(self.levels//2-1), self.levels//2-1)
        c2 = (1+math.exp(-self.c1))/(1-math.exp(-self.c1))
        return 2*c2*torch.sigmoid(self.c1*(x-2*level))-c2+2*level


class ScaledSigmoid(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == 'adaptive_alpha':
            self.alpha = nn.Parameter(torch.tensor(4.0))
        else:
            self.register_buffer('alpha', torch.tensor(4.0 if kind == 'fixed_alpha' else 1.0))

    def forward(self, x):
        return torch.sigmoid(self.alpha * x)


class ResearchNet(nn.Module):
    def __init__(self, inputs, hidden, outputs, activation, levels):
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.activation = (StableMultilevel(c1=6, levels=levels) if activation == 'multilevel'
                           else ScaledSigmoid(activation))
        self.fc2 = nn.Linear(hidden, outputs)
        self.output_activation = (StableMultilevel(c1=6, levels=levels)
                                  if activation == 'multilevel' else nn.Identity())

    def forward(self, x):
        return self.output_activation(self.fc2(self.activation(self.fc1(x))))


def fit_nn(x, y, vx, vy, cfg, seed, kind, order):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-8).astype(np.float32)
    xt = torch.from_numpy(np.asarray((x-mean)/scale, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    xv = torch.from_numpy(np.asarray((vx-mean)/scale, dtype=np.float32))
    yv = torch.from_numpy(np.asarray(vy, dtype=np.float32))
    levels = 2 if order == 2 else int(math.sqrt(order))
    model = ResearchNet(x.shape[1], int(cfg['nn_hidden_size']), y.shape[1], kind, levels)
    alpha_parameters = [p for name, p in model.named_parameters() if name == 'activation.alpha']
    other = [p for name, p in model.named_parameters() if name != 'activation.alpha']
    groups = [dict(params=other, lr=cfg['nn_learning_rate'])]
    if alpha_parameters:
        groups.append(dict(params=alpha_parameters, lr=cfg['alpha_learning_rate']))
    optimizer = torch.optim.Adam(groups)
    history, best, best_iteration, best_state = [], math.inf, 0, None

    def capture(iteration, training_loss, phase):
        nonlocal best, best_iteration, best_state
        with torch.no_grad():
            loss = nn.functional.mse_loss(model(xv), yv).item()
        alpha = float(model.activation.alpha.detach()) if hasattr(model.activation, 'alpha') else 0.0
        if not math.isfinite(loss):
            raise ValueError('Nonfinite neural validation loss')
        history.append((iteration, training_loss, loss, alpha, phase))
        if loss < best:
            best, best_iteration = loss, iteration
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    for epoch in range(1, int(cfg['nn_epochs'])+1):
        total = 0.0
        for indices in torch.randperm(len(xt)).split(int(cfg['nn_batch_size'])):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xt[indices]), yt[indices])
            loss.backward()
            optimizer.step()
            total += loss.item()*len(indices)
        capture(epoch, total/len(xt), 0)
        if epoch-best_iteration >= cfg['nn_patience']:
            break
    adam_epochs = len(history)
    model.load_state_dict(best_state)
    # Full-batch polishing uses training loss only; retain a change only if it
    # improves validation MSE. Test targets are never passed into this function.
    polish = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=10,
                               line_search_fn='strong_wolfe')
    for step in range(int(cfg['nn_lbfgs_steps'])):
        def closure():
            polish.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xt), yt)
            loss.backward()
            return loss
        polish.step(closure)
        with torch.no_grad():
            training_loss = nn.functional.mse_loss(model(xt), yt).item()
        capture(adam_epochs+step+1, training_loss, 1)
    model.load_state_dict(best_state)
    model.eval()
    info = dict(activation=kind, hidden_size=int(cfg['nn_hidden_size']),
                best_iteration=best_iteration, best_validation_mse=best,
                adam_epochs=adam_epochs, adam_budget_reached=adam_epochs == cfg['nn_epochs'],
                lbfgs_steps=int(cfg['nn_lbfgs_steps']),
                alpha=float(model.activation.alpha.detach()) if hasattr(model.activation, 'alpha') else None)
    return model, mean, scale, np.asarray(history), info


def load_dataset(path):
    data = loadmat(path)
    cfg = json.loads(str(data['configuration_json'].item()))
    condition = json.loads(str(data['condition_json'].item()))
    ideal = data['ideal_symbols'].ravel()
    bits = data['ideal_bits'].astype(np.int64)
    users = data['Y_train'].shape[1]//2
    inputs = data['X_train'].shape[1]
    if len(ideal) != condition['ModOrder'] or bits.shape != (len(ideal), int(math.log2(len(ideal)))):
        raise ValueError('Invalid constellation labels')
    if not np.isin(bits, (0, 1)).all() or len(np.unique(bits, axis=0)) != len(ideal):
        raise ValueError('Invalid bit mapping')
    for split in ('train', 'validation', 'test'):
        x, y = data['X_'+split], data['Y_'+split]
        if x.shape != (len(y), inputs) or y.shape[1] != 2*users or len(y) < 2:
            raise ValueError('Inconsistent dataset dimensions')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite dataset')
        target = y[:, 0::2] + 1j*y[:, 1::2]
        if np.min(abs(target[..., None]-ideal), axis=-1).max() > 1e-6:
            raise ValueError('Target is not a supplied QAM symbol')
    return data, cfg, condition, ideal, bits, users


def run(path, output):
    torch.set_num_threads(1)
    data, cfg, condition, ideal, bits, users = load_dataset(path)
    seed = int(data['seed'].item())
    output.mkdir(parents=True, exist_ok=True)
    x, y = data['X_train'], data['Y_train']
    vx, vy, tx = data['X_validation'], data['Y_validation'], data['X_test']
    ref_y = data['Y_test']
    reference = ref_y[:, 0::2].astype(float) + 1j*ref_y[:, 1::2]
    names, predictions, rows, fits, skipped = [], [], [], {}, []

    def record(name, prediction, parameters=0, multiplications=0, activations=0):
        if not np.iscomplexobj(prediction):
            prediction = prediction[:, 0::2].astype(float) + 1j*prediction[:, 1::2]
        if prediction.shape != reference.shape:
            raise ValueError(f'{name}: invalid prediction shape')
        names.append(name); predictions.append(prediction)
        user_indices = list(range(1, users+1)) + ([0] if users > 1 else [])
        for user in user_indices:
            pr = prediction.ravel(order='F') if user == 0 else prediction[:, user-1]
            target = reference.ravel(order='F') if user == 0 else reference[:, user-1]
            row = dict(method=name, user=user, **metrics(pr, target, ideal, bits),
                       parameters=parameters, real_multiplications=multiplications,
                       activation_evaluations=activations)
            rows.append(row)
            if user == (0 if users > 1 else 1):
                print(f"{name:24s} EVM={row['evm_percent']:7.3f}% BER={row['ber']:.5g}", flush=True)

    baseline = data['baseline_test']
    record(cfg['baseline_name'], baseline.astype(np.complex128))
    cx = x[:, 0::2].astype(float) + 1j*x[:, 1::2]
    cy = y[:, 0::2].astype(float) + 1j*y[:, 1::2]
    ct = tx[:, 0::2].astype(float) + 1j*tx[:, 1::2]
    weights = np.linalg.lstsq(cx, cy, rcond=None)[0]
    record('Joint FFE' if users > 1 else 'Complex FFE', ct@weights, 2*weights.size, 4*weights.size)
    np.savez(output/'ffe.npz', weights=weights)
    for degree in cfg['polynomial_orders']:
        degree = int(degree)
        count = math.comb(x.shape[1]+degree, degree)-1
        if count > cfg['max_polynomial_features']:
            skipped.append(dict(model=f'Volterra {degree}', features=count,
                                reason='Predeclared polynomial feature budget'))
            continue
        model = PolynomialRegressor(degree).fit(x, y, vx, vy, cfg['ridge_grid'])
        name = 'Linear I/Q' if degree == 1 else f'Joint Volterra {degree}' if users > 1 else f'Volterra {degree}'
        record(name, model.predict(tx), model.parameter_count, model.multiplications)
        model.save(output/f'volterra_{degree}.npz')
        fits[name] = dict(ridge=model.ridge, rank=model.rank, features=count)
    activations = [cfg['activations']] if isinstance(cfg['activations'], str) else list(cfg['activations'])
    if condition['ActivationStudy']:
        activations += ['fixed_alpha', 'adaptive_alpha']
    for kind in dict.fromkeys(activations):
        model, mean, scale, history, info = fit_nn(x, y, vx, vy, cfg, seed, kind, condition['ModOrder'])
        with torch.no_grad():
            predicted = model(torch.from_numpy(np.asarray((tx-mean)/scale, dtype=np.float32))).numpy()
        name = ('Joint NN ' if users > 1 else 'NN ') + kind.replace('_', ' ')
        record(name, predicted, sum(p.numel() for p in model.parameters()),
               cfg['nn_hidden_size']*(x.shape[1]+y.shape[1]),
               cfg['nn_hidden_size'] + (y.shape[1] if kind == 'multilevel' else 0))
        torch.save(dict(state_dict=model.state_dict(), mean=mean, scale=scale, info=info,
                        inputs=x.shape[1], outputs=y.shape[1], mod_order=condition['ModOrder']),
                   output/f'nn_{kind}.pt')
        np.savetxt(output/f'history_{kind}.csv', history, delimiter=',',
                   header='iteration,training_mse,validation_mse,alpha,phase', comments='')
        fits[name] = info
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = dict(configuration=cfg, condition=condition, seed=seed, fits=fits, skipped=skipped,
                    dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__, torch=torch.__version__),
                    partition_samples={s:len(data['X_'+s]) for s in ('train','validation','test')})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    savemat(output/'results.mat', dict(method_names=np.asarray(names, dtype=object),
                                      predictions=np.stack(predictions, axis=2), reference=reference),
            do_compression=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.output)
