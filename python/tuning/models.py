"""Flexible Gray-bit neural equalizers; no hardware topology restriction."""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn


class Multilevel(nn.Module):
    """Shared smooth staircase; learned widths/locations change its shape.

    This is an experimental activation ablation inspired by multilevel
    sigmoids, not a claim of mathematical or literature novelty.
    """
    def __init__(self, learned=False):
        super().__init__()
        self.learned = learned
        centers = torch.tensor([-1.0, 0.0, 1.0])
        slopes = torch.full((3,), math.log(math.expm1(2.0)))
        if learned:
            self.centers = nn.Parameter(centers)
            self.raw_slopes = nn.Parameter(slopes)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('raw_slopes', slopes)

    def forward(self, x):
        slope = nn.functional.softplus(self.raw_slopes)+1e-4
        return (2*torch.sigmoid((x[..., None]-self.centers)*slope).sum(dim=-1)/3)-1


class BitEqualizer(nn.Module):
    def __init__(self, inputs, bits, widths, activation):
        super().__init__()
        factories = dict(tanh=nn.Tanh, silu=nn.SiLU, softsign=nn.Softsign,
                         multilevel=lambda: Multilevel(False),
                         learned_multilevel=lambda: Multilevel(True))
        layers = []
        for width in widths:
            layers.extend([nn.Linear(inputs, width), factories[activation]()])
            inputs = width
        layers.append(nn.Linear(inputs, bits))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def catalog():
    return [dict(widths=w, activation=a, lr=.003, weight_decay=1e-4)
            for w, a in [([4], 'tanh'), ([6], 'tanh'), ([8], 'tanh'), ([16], 'tanh'), ([24], 'silu'),
                         ([16, 16], 'tanh'), ([16], 'softsign'),
                         ([16], 'multilevel'), ([16], 'learned_multilevel')]]


def train(x, bits, vx, vbits, spec, *, seed=42, epochs=600, patience=60, polish=10):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    xt = torch.from_numpy(np.asarray((x-mean)/scale, dtype=np.float32))
    vt = torch.from_numpy(np.asarray((vx-mean)/scale, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(bits, dtype=np.float32))
    vy = torch.from_numpy(np.asarray(vbits, dtype=np.float32))
    net = BitEqualizer(x.shape[1], bits.shape[1], spec['widths'], spec['activation'])
    optimizer = torch.optim.AdamW(net.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    best, state, best_step, history = math.inf, None, 0, []

    def capture(step, loss, phase):
        nonlocal best, state, best_step
        with torch.no_grad():
            logits = net(vt)
            val = nn.functional.binary_cross_entropy_with_logits(logits, vy).item()
            ber = (logits.ge(0) != vy.bool()).float().mean().item()
        history.append([step, loss, val, ber, phase])
        if val < best:
            best, state, best_step = val, copy.deepcopy(net.state_dict()), step

    for epoch in range(1, epochs+1):
        total = 0.
        for batch in torch.randperm(len(xt)).split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt[batch]), yt[batch])
            loss.backward(); optimizer.step()
            total += loss.item()*len(batch)
        capture(epoch, total/len(xt), 0)
        if epoch-best_step >= patience:
            break
    adam_steps = len(history)
    net.load_state_dict(state)
    lbfgs = torch.optim.LBFGS(net.parameters(), lr=.5, max_iter=10, line_search_fn='strong_wolfe')
    for step in range(polish):
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt), yt)
            loss.backward()
            return loss
        loss = lbfgs.step(closure)
        capture(adam_steps+step+1, loss.item(), 1)
    net.load_state_dict(state); net.eval()
    dims = [x.shape[1], *spec['widths'], bits.shape[1]]
    hidden = sum(spec['widths'])
    nonlinear = hidden*(3 if 'multilevel' in spec['activation'] else 1)
    dense = sum(a*b for a,b in zip(dims[:-1], dims[1:]))
    extra = 4*hidden if 'multilevel' in spec['activation'] else hidden if spec['activation']=='silu' else 0
    info = dict(**spec, input_size=x.shape[1], output_bits=bits.shape[1],
                parameters=sum(p.numel() for p in net.parameters()),
                multiplications=dense+extra, dense_multiplications=dense,
                activation_multiplications=extra,
                nonlinear_evaluations=nonlinear, comparisons=bits.shape[1],
                best_step=best_step, validation_bce=best, epochs_completed=adam_steps,
                polish_steps=polish, seed=seed)
    return dict(net=net, mean=mean, scale=scale, info=info, history=np.asarray(history))


def predict(model, x):
    with torch.no_grad():
        value = torch.from_numpy(np.asarray((x-model['mean'])/model['scale'], dtype=np.float32))
        return model['net'](value).numpy()


def save(model, path):
    torch.save(dict(state_dict=model['net'].state_dict(), mean=torch.from_numpy(model['mean']),
                    scale=torch.from_numpy(model['scale']), info=model['info']), path)
