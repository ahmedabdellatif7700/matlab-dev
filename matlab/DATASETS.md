# Equalizer dataset export

This is step 4 of the agreed research sequence, on `codex/dataset-export`, branched from physical-link commit `a2b9c6f`. The exporter repeats the existing physical-stage functions; `+arof/` remains stages only. It does not train an equalizer or change the fitted physical parameters.

From the repository root:

```matlab
addpath('matlab');
cfg = link_configuration();
opts = dataset_configuration();
folder = export_datasets(cfg, opts);
```

For a small execution check, use `dataset_configuration("smoke")`. Output goes to a new run under `datasets/physical_link/equalizer_datasets/`. Existing folders are never overwritten. Data are generated locally and excluded from Git; the export code and compact run/validation records are committed.

## Independent sequences

| ROP (dBm) | Training seed | Validation seed | Test seed |
|---:|---:|---:|---:|
| -22 | 9101 | 9201 | 9301 |
| -20 | 9102 | 9202 | 9302 |
| -17.5 | 9103 | 9203 | 9303 |

Each power directory contains separate `train.mat`, `validation.mat` and `test.mat` files. The study exports 32,768 training, 16,384 validation and 65,536 test payload symbols per power. Smoke lengths are 2,048 / 1,024 / 4,096. Noise seeds are symbol seeds plus 900,000; all symbol and noise seeds are distinct. No waveform or tap window crosses a split boundary.

These powers cover the low, middle and high ends of the established ROP range. They are selected before comparing equalizers. Each receiver must consume the same corresponding files. Use training for coefficient updates, validation for tuning and final test for frozen-choice evaluation. Never concatenate the files and invoke the historical Python 80/20 split.

## MAT schema

MAT files use `-v7`, readable by `scipy.io.loadmat`.

| Variable | Shape / meaning |
|---|---|
| `X` | N x 10 decoded Q8.24 inputs, stored as exact float64 multiples of 2^-24. |
| `Y` | N x 2 continuous transmitted I/Q targets, decoded Q8.24. |
| `X_q8_24`, `Y_q8_24` | Same shapes, signed int32 Q8.24 words; authoritative NN inputs/targets. |
| `rx_iq`, `tx_iq` | Unnormalized float64 physical samples, prefix followed by payload; no outer guards. Retained for diagnostics and future tap choices. |
| `target_indices` | One-based positions of the N payload targets within the physical sample arrays. |
| `is_receiver_prefix` | Marks the 1,024 known symbols used only for receiver timing/gain calibration. |
| `is_equalizer_training` | True only for training-file payload symbols. False for prefix, validation and test. |
| `metadata` | Seeds, split, dimensions, tap offsets, normalization, receiver gain/timing, metrics, format and saturation counts. |
| `configuration` | Actual physical-link settings used for that file, including waveform drive scaling. |

Default feature order is `[I(t), Q(t), I(t-1), Q(t-1), ..., I(t-4), Q(t-4)]`. This is a configurable observation window, not a selected NN architecture. `opts.tap_offsets_symbols` can change it before exporting. Only current/past received symbols are used. At the first payload row, history may come from that file's own known prefix. Prefix symbols are never exported as `X`/`Y` targets.

Both inputs and targets are divided by the theoretical RMS of the known Gray-QAM constellation (`sqrt(10)` for the current unnormalized 16-QAM). No training, validation or test sample statistics determine this scale. Receiver timing and gain are independently estimated using only each file's known prefix, as in the existing physical link. Convert normalized estimates back by `metadata.normalization_scale` before ordinary QAM slicing/EVM against the unnormalized physical targets.

## Q8.24 contract and Python use

At import, quantization is nearest with ties to even, then signed 32-bit saturation. This matches `to_q` in the user-supplied `nn_training_model/hardware/training/fixedpoint.py`. That engine's internal `round_shift` uses ties away from zero; the exporter does not change the engine or its arithmetic stages. Exported saturation counts must be reviewed; the validated smoke/study records report them explicitly.

```python
from scipy.io import loadmat

data = loadmat("path/to/rop_-20.0_dbm/train.mat")
X = data["X"]                    # keep float64: exact decoded Q8.24
Y = data["Y"]
X_words = data["X_q8_24"]       # int32, direct fixed-point-engine inputs
Y_words = data["Y_q8_24"]
```

The supplied engine's `to_q(float(X[i, j]))` reproduces `X_words[i, j]` exactly. Do not cast decoded arrays to float32 before Q8.24 import; that can discard low bits. The historical Python loader casts to float32 and resplits one file, so it is not yet the training entry point for these datasets. Integration with the supplied engine, configurable NN dimensions/activations and golden RTL stages remains the agreed later step. This export includes no Q8.8 or mixed-precision NN representation; raw float64 arrays describe the physical link only.

`manifest.csv` lists each file, split, seeds, sample counts, dimensions, unequalized bit counts/EVM and saturation counts. `export_configuration.json` records link and export settings. Full-study Python validation compares all words against the actual supplied `to_q`, reconstructs tap alignment, checks masks/seeds and records MAT SHA-256 hashes. This validates the export contract, not an equalizer advantage or independent experimental replication.

[Completed study-run manifest and validation record](results/dataset_export/README.md).
