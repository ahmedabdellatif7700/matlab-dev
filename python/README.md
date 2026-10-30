# Adaptive I/Q equalizers

The current experiment compares LMS FFE with compact ANNs on MATLAB-generated data from one millimeter-wave analog radio-over-fiber link. Use the existing `equalizer.py` entry point:

```sh
python python/equalizer.py --workflow adaptive --run /path/to/MATLAB/comparison/export
```

MATLAB first runs `compare_adaptive_equalizers('study')`; see [the experiment methodology](../matlab/ADAPTIVE_COMPARISON.md) for physical assumptions, dataset splits and complete commands. The physical `matlab/main.m` remains the explicit stage runner. It does not invoke Python automatically.

- `config.py: ADAPTIVE` defines epochs, LMS tap counts/rates, ANN widths/activations/rates and initialization seed.
- `adaptive_comparison.py` consumes MATLAB train/validation/test MAT files, trains receivers, selects by validation EVM, freezes test inference, and writes metrics, model words and plots.
- `ffe_equalizer.py: Q824LMS` implements the two-output real/widely-linear LMS FFE with bias.
- `hardware/training/` contains the supplied training model and extensions. Its forward, loss, pre-update-weight backward and immediate SGD-update stages are retained.

All stored values in this workflow use Q8.24. ANN outputs are continuous I/Q; BER uses square Gray-QAM slicing and EVM uses the I/Q error. Eight chronological passes through the training record are an offline adaptation experiment, not a live FPGA throughput demonstration. New widths and tanh are not RTL-qualified.

Independent jobs can use `--only reference`, `--only bias_offset` or `--only driver_compression`; merge their reports with `--phase report`. Candidate checkpoints support resuming an unchanged configuration. Large datasets, checkpoints and predictions stay in ignored run directories. Compact completed results are curated under `matlab/results/`.

## Historical workflows

The older `legacy` and `research` dispatch modes, `tuning/`, ONNX and Q8.8 inference utilities remain for provenance. They are not part of the current experiment and their historical artifact files are not Q8.24 results. The removed MATLAB scenario runners cannot be used to launch them from the current physical baseline. Use the explicit `--workflow adaptive` command above.

The adaptive workflow also writes side-by-side I/Q constellations in `constellations/` for every condition/power. `--phase report` regenerates these from the saved predictions and MATLAB test files without retraining. [Saved gallery](../matlab/results/adaptive_equalizers/constellations/README.md).

## Equalizer tuning

Use `--preset tuned` for the 32-pass, scheduled-learning-rate refinement in `config.py: ADAPTIVE_TUNED`. The eight-pass baseline remains available unchanged. [Tuning protocol and fresh-test commands](../matlab/EQUALIZER_TUNING.md). With `--phase report --baseline-run <original-run>`, previously frozen LMS/ANN weights are evaluated on the new test records to produce paired before/after metrics and curves.
