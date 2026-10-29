# Physical stages and BER versus received optical power

Run `matlab/main.m`. It connects the physical stages in `matlab/+arof/` explicitly, then calls `ber_vs_rop.m` to simulate the unequalized BER curve using fixed link parameters. Set physical parameters and sweep lengths in `matlab/link_configuration.m`. [Saved simulation-only BER curve](matlab/results/ber_vs_rop/README.md).

```matlab
addpath('matlab');
cfg = link_configuration();
folder = ber_vs_rop(cfg);
```

`+arof/` contains only physical-stage functions. The shared FFT filter is an implementation helper in `+arof/private/`. There are no active paper-comparison, parameter-fitting or verification runners. The received-power curve uses the previously fitted effective gain and noise values without refitting. The earlier partial-match limitations remain recorded in [the historical report](matlab/results/received_power/README.md).

The physical model and assumptions are documented in [matlab/PHYSICAL_LINK.md](matlab/PHYSICAL_LINK.md); the research sequence is in [METHODOLOGY.md](METHODOLOGY.md). MATLAB requires Communications Toolbox and Signal Processing Toolbox. Python is not invoked by the current physical-link runner.

Older MATLAB scenario suites, archived implementations and their runners have been removed from the working tree. Their source remains in Git history. Existing reference data and result records are retained for provenance; they are not dependencies of the current runner.

NN development will use the supplied `nn_training_model/` package in the later research phase, preserving its training stages and Q8.24 arithmetic. No new equalizer comparison is included in this physical-link baseline.

Dataset export for the next research phase is available on `codex/dataset-export`: run `export_datasets(link_configuration(), dataset_configuration())` after adding `matlab` to the MATLAB path. Training, validation and test sequences are independent. See [the dataset contract](matlab/DATASETS.md).
