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

The next experiment compares LMS FFE and ANN using the supplied training model's stages and Q8.24 arithmetic. Run `compare_adaptive_equalizers('study')` in MATLAB, then `python python/equalizer.py --workflow adaptive --run <printed-directory>`. [Physical conditions, selection method and limitations](matlab/ADAPTIVE_COMPARISON.md); [Python entry point](python/README.md). The standalone physical-stage baseline remains available unchanged.

Completed LMS/ANN comparison: [BER/EVM figures, parameter tables and limitations](matlab/results/adaptive_equalizers/README.md). The selected ANN lowers observed BER at all nine points; the compressed-driver high-power result has only two versus four errors and needs longer independent confirmation.
