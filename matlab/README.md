# Current A-RoF physical link

Run `main.m` in MATLAB, or from the repository root:

```matlab
run('matlab/main.m')
```

The required code consists of:

- `main.m`: explicit physical-stage connections.
- `link_configuration.m`: physical parameters, fixed gain/noise values and sweep settings.
- `ber_vs_rop.m`: simulation-only BER-versus-received-optical-power curve.
- `+arof/`: physical-stage functions, with a private FFT-filter implementation.

For a short curve run:

```matlab
addpath('matlab');
cfg = link_configuration();
cfg.sweep_seeds = 8701;
cfg.sweep_payload_symbols = 4096;
folder = ber_vs_rop(cfg);
```

Results are written under `datasets/physical_link/ber_vs_rop/`. [The saved full curve](results/ber_vs_rop/README.md) reuses the previously completed five-seed simulation. [PHYSICAL_LINK.md](PHYSICAL_LINK.md) records physical parameters and assumptions. Reference readings and historical results are retained for provenance; the runtime does not read them.

Legacy scenario code, archived implementations and old runners have been removed. Their source remains in Git history. MATLAB needs Communications Toolbox and Signal Processing Toolbox. The current physical-link runner does not invoke Python.

## Adaptive equalizer experiment

`compare_adaptive_equalizers.m` reuses the physical stages for three declared component conditions. `dataset_configuration.m` and `export_datasets.m` write independent train/validation/test MAT files for Python. These are experiment/export functions outside `+arof/`; `main.m` remains the physical-stage runner. See [ADAPTIVE_COMPARISON.md](ADAPTIVE_COMPARISON.md) and [DATASETS.md](DATASETS.md).
