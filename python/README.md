# Python equalizer workflow

MATLAB `main.m` selects among five impairment studies. Scenario 1 uses
`scenario_benchmark.py`; Scenarios 2–5 use `research_suite.py`. See
[the scenario index](../matlab/SCENARIOS.md) for physical models, fair
FFE/Volterra/NN comparisons, plots, and validation. Each study has separate
per-run artifacts and does not change the hardware training/inference
workflow below. Run `matlab/legacy_main.m` for the original ONNX/Q8.8 path.
`archive_suite.py` retains compact completed-study figures, metrics and
provenance without checking large waveform datasets into Git.

- `equalizer.py`: main MATLAB-driven off-chip training/inference script.
- `config.py`: single source of configuration for the unified 4–12–2 architecture, activation choice, optimizer, batch size, shuffling, and output paths.
- `activations/`: supported activation functions.
- `hardware/q8_8_inference_arithmetic.py`: fixed-point inference calculations.
- `hardware/q8_24_training_arithmetic.py`: proposed fixed-point SGD calculations (training RTL remains under development).
- `software/software_inference.py`: reads exported Q8.8 memories and regenerates `software_outputs.csv`; optionally compares a hardware CSV. **Does not read MATLAB `.mat` files.**
- `software/software_training.py`: checks existing FP32 SGD training artifacts. It does not train the network.

## Generated artifacts

`artifacts/models/`: `equalizer.onnx`, `equalizer_q8_8.npz`

`artifacts/inference/`: MATLAB-generated `test_inputs.mem`, Python-generated `parameters.mem` and `software_outputs.csv`

`artifacts/training/`: reserved for on-chip training reference/export outputs; to be finalized separately.

MATLAB writes `../datasets/training_data.mat` and `artifacts/inference/test_inputs.mem`. Python exports `../datasets/equalized_output_onnx_fp32.mat` and `../datasets/equalized_output_software_q8_8.mat`, while C++ exports `../datasets/equalized_output_fpga.mat`. `equalizer.py` consumes the canonical MATLAB input words and writes `parameters.mem` and `software_outputs.csv`. The `software_inference.py` utility reads `parameters.mem` and `test_inputs.mem` to regenerate the CSV; the combined parameter memory contains 86 words of weights/biases followed by 42 zero-filled reserved words. The legacy individual weight/bias `.mem` files are no longer exported.

Run the main pipeline using `make run` (requires the configured Conda environment, ONNX, and ONNX Runtime). Run the independent software inference check using `make software-inference`. Use `make software-training` only after training artifacts exist.

### Console log

`make run` displays the console output and writes the same output, including errors, to `python/output.txt` regardless of the shell working directory. The log is overwritten on each run.

## Unified training configuration

Configure the experiment in `config.py`. The default is `MODE="real"`, `ACTIVATION="sigmoid"`, `HIDDEN_SIZE=12`, `OPTIMIZER="sgd"`, `BATCH_SIZE=1`, `SHUFFLE=False`, and `LEARNING_RATE=0.001`. This is **online-style SGD in Python**, not proof of real-time FPGA training. The three current activations are `sigmoid`, `variable_alpha`, and `stable_alpha`; each uses the same 4–12–2 network and optimizer. The variable-alpha parameter is trainable and participates in the same optimizer updates.

Every training run overwrites the Python outputs in `artifacts/models/` and
`artifacts/inference/` regardless of the activation. The Python exports are
`equalizer.onnx`, `equalizer_q8_8.npz`, `parameters.mem`, and
`software_outputs.csv`; MATLAB independently exports `test_inputs.mem`. There are no activation-specific suffixes. To
compare experiments, copy the generated files and console log elsewhere
before running the next configuration.

The packed 128-word `parameters.mem` stores the 86 weights and biases of the
4–12–2 network (plus 42 zero-filled reserved words). For alpha-scaled
activations, the additional Q8.8 alpha coefficient is stored in
`equalizer_q8_8.npz`, not in the packed memory image. Q8.8 software inference
uses the custom Python activation plus Q8.8-rounded MAC operations; the
current RTL still supports **standard sigmoid only**, so non-sigmoid CSVs
must **not** be treated as bit-exact FPGA predictions. The independent
`software/software_inference.py` reads the saved alpha from the NPZ for those
experiments and regenerates the same software CSV.

Switching from Adam/minibatches to SGD/one sample changes learned weights and may change convergence. Previous saved artifact files in this ZIP represent the **old Adam baseline** until `make run` regenerates them; do not interpret these existing artifacts as new SGD results.


### Canonical Q8.8 inference inputs

MATLAB `main.m` writes `artifacts/inference/test_inputs.mem` from the final
20% of `X` before starting Python training. This file contains 4 hexadecimal
16-bit Q8.8 words per sample, in `[I[n], Q[n], I[n-1], Q[n-1]]` order.
Python ONNX FP32 inference still reads the original `training_data.mat` data.
Python Q8.8 inference reads MATLAB's `test_inputs.mem` and Python's
`parameters.mem` **without re-quantizing the inference inputs**. The latter
contains 86 signed Q8.8 parameters and 42 zero-reserved words. MATLAB must
be run first to create the dataset and memory file. `make clean` preserves
MATLAB's `test_inputs.mem`.

## Flexible bitwise equalizer design

`../matlab/main.m` now orchestrates discovery and independent confirmation through the existing `equalizer.py` entry point, with experiment orchestration in `tuning/`. Network widths and activations are unrestricted by the legacy FPGA 4–12–2 topology. The search catalog includes tanh, SiLU, softsign and fixed/learned smooth multilevel activations. No scikit-learn dependency is needed.

Run `python -B equalizer.py --run <generated-run-directory> --phase discovery`, then `--phase confirmation` only after MATLAB creates fresh confirmation data, followed by `--phase report`. Discovery can resume completed candidates. Read `../matlab/DESIGN_STUDY.md` for target locking, conventional controls, resource accounting and limitations. Legacy quantized inference is a separate implementation; the new arbitrary bit networks are floating point and have not been exported to FPGA.

Set widths and activation candidates in `config.py`; the shared model is `RealEqualizer` in `equalizer.py`, and activation implementations are in `activations/`. The no-argument legacy MATLAB launch and its parameter names are preserved. Research data uses MAT `X`/`Y` arrays plus split counts. No separate verification entry point is required.
