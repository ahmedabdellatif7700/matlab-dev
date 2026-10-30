# Equalizer tuning on the unchanged physical link

Run `adaptive_tuned_01` refines receiver parameters only. Physical component settings, 400 mVpp input, ROP points, effective gain/noise, 16-QAM and the 40 GSa/s numerical grid are unchanged. Training/validation Q8.24 arrays match the original experiment exactly. Each final-test symbol/noise seed is shifted by 40,000. [Preset, methodology and commands](../../EQUALIZER_TUNING.md).

The reference engine retains Q8.24 forward → MSE → backward with pre-update weights → sample-by-sample SGD. The search increases the epoch budget from 8 to 32, reduces the learning rate after epochs 16 and 24, tests two/three ANN symbol taps and 12/16 hidden units, and strengthens the LMS learning-rate search. Each candidate receives 262,144 training updates; validation EVM selects its checkpoint and the primary ANN. All model working states fit 256 reference RAM words; this excludes storage of the full training record and is not RTL qualification.

## Paired before/after comparison

**Old frozen models and tuned models are evaluated on identical fresh test symbols.** Each row has 65,536 symbols / 262,144 bits. These are not comparisons against old test-sequence metrics.

| Condition | ROP (dBm) | Old ANN errors | Tuned ANN errors | Old ANN BER | Tuned ANN BER | Old ANN EVM (%) | Tuned ANN EVM (%) | Tuned LMS BER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | -22 | 3386 | 3318 | 0.0129166 | 0.0126572 | 19.719 | 19.390 | 0.0144119 |
| reference | -20 | 318 | 261 | 0.00121307 | 0.000995636 | 14.634 | 14.147 | 0.00206757 |
| reference | -17.5 | 49 | 24 | 0.00018692 | 9.15527e-05 | 11.155 | 10.615 | 0.000453949 |
| bias_offset | -22 | 3362 | 3217 | 0.012825 | 0.0122719 | 19.877 | 19.476 | 0.0161362 |
| bias_offset | -20 | 532 | 345 | 0.00202942 | 0.00131607 | 15.296 | 14.639 | 0.00396729 |
| bias_offset | -17.5 | 105 | 52 | 0.000400543 | 0.000198364 | 12.808 | 12.050 | 0.000915527 |
| driver_compression | -22 | 3012 | 3032 | 0.0114899 | 0.0115662 | 19.460 | 19.140 | 0.0125275 |
| driver_compression | -20 | 174 | 161 | 0.000663757 | 0.000614166 | 13.861 | 13.526 | 0.00102615 |
| driver_compression | -17.5 | 5 | 4 | 1.90735e-05 | 1.52588e-05 | 10.132 | 9.713 | 3.05176e-05 |

Tuning lowers observed ANN EVM at all nine points and observed ANN BER at eight. At the low-power driver-compression point, BER rises slightly (3,012 to 3,032 errors) despite lower EVM; a validation-MSE improvement does not guarantee a BER improvement. At -17.5 dBm, reference ANN errors fall from 49 to 24 and bias-offset ANN errors from 105 to 52 on the same fresh records. These are conditional single-seed results.

The complete paired metrics, including old/tuned LMS EVM and nominal Wilson intervals, are in `before_after_test_metrics.csv`. BER differences with only a handful of errors do not establish a reliable gain. Zero-error observations are not proof of zero true BER; plots mark them with open triangles at the nominal 95% Wilson upper endpoint. No improvement is inferred solely from a sharper-looking displayed subset.

## Selected neural parameters

| Condition | ROP (dBm) | Architecture | Activation | Initial learning rate | Selected epoch | Parameters |
|---|---:|---|---|---:|---:|---:|
| reference | -22 | 4-16-2 | tanh | 0.008 | 32 | 114 |
| reference | -20 | 4-16-2 | tanh | 0.008 | 32 | 114 |
| reference | -17.5 | 4-16-2 | tanh | 0.008 | 32 | 114 |
| bias_offset | -22 | 4-16-2 | tanh | 0.008 | 32 | 114 |
| bias_offset | -20 | 4-16-2 | tanh | 0.008 | 32 | 114 |
| bias_offset | -17.5 | 4-16-2 | tanh | 0.008 | 32 | 114 |
| driver_compression | -22 | 4-16-2 | tanh | 0.008 | 32 | 114 |
| driver_compression | -20 | 4-16-2 | tanh | 0.008 | 32 | 114 |
| driver_compression | -17.5 | 4-16-2 | tanh | 0.008 | 32 | 114 |

Learning-rate multipliers are 1 for epochs 1–16, 0.5 for 17–24 and 0.25 for 25–32; the engine stores each effective rate as a Q8.24 word. 9/9 selected primary ANN checkpoints are at the final epoch, so the experiment does not establish full optimization convergence. The comparison is budget-limited. `learning_curves.csv` retains every epoch, and `selected_parameters.csv` contains all activation/width groups and tuned LMS settings. Rates/history may differ between activations, so this is not an isolated activation ablation.

Recorded fixed-point saturation events across all 153 fits: 0.

## Plots and constellations

- Paired before/after [BER](before_after_ber_comparison.png) / [PDF](before_after_ber_comparison.pdf) and [EVM](before_after_evm_percent_comparison.png) / [PDF](before_after_evm_percent_comparison.pdf).
- Paired constellation diagrams (same fresh symbols): [reference at -17.5 dBm](before_after_constellations/reference_-17.5_dbm.png), [bias_offset at -17.5 dBm](before_after_constellations/bias_offset_-17.5_dbm.png), [driver_compression at -17.5 dBm](before_after_constellations/driver_compression_-17.5_dbm.png).
- All tuned receiver groups: [BER](ber_comparison.png) and [EVM](evm_percent_comparison.png).
- Condition reports, parameter tables and side-by-side unequalized/LMS/ANN constellations: [reference](reference/README.md), [bias offset](bias_offset/README.md), [driver compression](driver_compression/README.md).

Constellations show the same first 8,192 held-out symbols in each panel, with ideal QAM crosses and common axes. Annotations use the full test set. There is no hard-decision snapping, extra gain/phase correction or selected-symbol filtering.

## Provenance and limits

The physical-stage runner `matlab/main.m` is unchanged. Per-condition export configuration/manifests and `dataset_checks.json` preserve all physical settings, seeds and MAT hashes. `implementation_checks.json` records original-engine phase equivalence, the Q8.24 learning-rate schedule, extended-history execution, baseline state restoration and syntax/Code Analyzer checks. Old state restoration reproduces the original nine operating points' error counts and EVM before evaluating fresh sequences. Large MAT datasets, checkpoints and int32 prediction/state files remain in ignored run directories.

This is offline repeated training on known symbols, not a measured live on-chip adaptation experiment. Extended configurations require RTL verification and KV260 resource/throughput/power measurements. The physical model retains the earlier partial effective calibration; one fresh test sequence and one ANN initialization per point do not establish multiseed robustness or experimental-device performance. Nominal Wilson intervals assume independent bits. A tuned-ANN advantage over LMS does not establish superiority over Volterra or uniquely necessary neural equalization.
