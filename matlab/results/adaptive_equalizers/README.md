# Q8.24 LMS FFE and ANN comparison

Completed run `adaptive_40g_01` on `codex/development`. MATLAB generates one physical mmWave A-RoF link; Python trains and infers using the supplied fixed-point stages. The standalone physical-stage `matlab/main.m` is unchanged. [Protocol and physical assumptions](../../ADAPTIVE_COMPARISON.md).

## Conditions and selected receiver

| Condition | Component change | ROP points (dBm) |
|---|---|---|
| Reference | 2.20 V MZM bias, linear IF driver | -22, -20, -17.5 |
| Bias offset | MZM bias 2.42 V (+0.05 Vpi, static) | -22, -20, -17.5 |
| Driver compression | Rapp IF-envelope driver: saturation 3.5 V, smoothness 2 | -22, -20, -17.5 |

The topology, 400 mVpp input, 16-QAM, 0.4 Gbaud, 15 km fiber, 1 m wireless, effective gain and aggregate noise PSD remain fixed. Component sensitivity values are engineering assumptions, not measured paper specifications. A 40 GSa/s internal grid passes short-record 40/80 GSa/s convergence checks for all conditions. The physical AWG remains 4 GSa/s.

Validation EVM selects a **4-16-2 tanh ANN**, 114 parameters, learning rate 0.004 (Q8.24 word 67109), at epoch 8 for all nine points. Each candidate receives 8,192 training symbols, eight chronological passes and 4,096 separate validation symbols. LMS chooses among 2/5/9 complex-symbol taps and three learning rates; it is a two-output real/widely-linear FIR with bias. Selected LMS models have 10, 22 or 38 parameters. There is no matched inference-budget claim. Final test has 65,536 independent payload symbols per operating point.

## Frozen-test results

| Condition | ROP (dBm) | LMS errors | ANN errors | LMS BER | ANN BER | LMS EVM (%) | ANN EVM (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | -22 | 3644 | 3299 | 0.0139008 | 0.0125847 | 20.540 | 19.736 |
| reference | -20 | 535 | 320 | 0.00204086 | 0.0012207 | 14.723 | 14.548 |
| reference | -17.5 | 73 | 42 | 0.000278473 | 0.000160217 | 11.116 | 11.035 |
| bias_offset | -22 | 4351 | 3506 | 0.0165977 | 0.0133743 | 20.920 | 19.935 |
| bias_offset | -20 | 909 | 454 | 0.00346756 | 0.00173187 | 16.113 | 15.510 |
| bias_offset | -17.5 | 156 | 69 | 0.000595093 | 0.000263214 | 13.235 | 12.756 |
| driver_compression | -22 | 3149 | 2914 | 0.0120125 | 0.011116 | 19.918 | 19.277 |
| driver_compression | -20 | 239 | 164 | 0.000911713 | 0.00062561 | 13.963 | 13.833 |
| driver_compression | -17.5 | 4 | 2 | 1.52588e-05 | 7.62939e-06 | 10.077 | 10.194 |

Every error count uses 262,144 bits. The selected ANN has lower observed BER at all nine points; EVM is lower at eight. The strongest current bias-offset comparison at -17.5 dBm is 156 LMS errors versus 69 ANN errors (55.8% lower observed BER). These findings are conditional on the model, finite training budget and this one test sequence/initialization per point.

**Do not treat the compressed-driver high-power point as an established BER gain:** it has only four LMS errors versus two ANN errors, with overlapping nominal Wilson intervals; LMS also has slightly lower EVM there. Longer independent records are needed. Compression before a nonlinear MZM can soften the cascade rather than make it worse; this condition has lower unequalized BER than the reference in the current runs. Distinct condition seeds also prevent interpreting small cross-condition differences as paired measurements.

## Activation comparison

At -20 dBm, separately validation-tuned 4-12-2 receivers give:

| Condition | Sigmoid BER | Tanh BER | Sigmoid EVM (%) | Tanh EVM (%) |
|---|---:|---:|---:|---:|
| reference | 0.00123978 | 0.00124359 | 14.843 | 14.652 |
| bias_offset | 0.00211716 | 0.00177765 | 15.711 | 15.720 |
| driver_compression | 0.000617981 | 0.000652313 | 14.079 | 13.996 |

The 12-unit models share architecture, initialization seed, data order and update budget; each activation has its own validation-selected learning rate. Tanh is not uniformly better in test BER. The 16-unit tanh result adds a width change, so its advantage is not an isolated activation contribution. Selected ANN checkpoints reach the final allowed epoch; these are budget-limited results, not proof that training has converged.

## Artifacts and validation

- [Combined BER plot](ber_comparison.png), [PDF](ber_comparison.pdf); [combined EVM plot](evm_percent_comparison.png), [PDF](evm_percent_comparison.pdf).
- Per-condition plots and parameter tables: [reference](reference/README.md), [bias offset](bias_offset/README.md), [driver compression](driver_compression/README.md).
- `test_metrics.csv`: all receiver groups, raw counts, nominal Wilson intervals and the validation-selected primary flag. No test result selects a model.
- `selected_parameters.csv`: dimensions, activations, float/quantized learning rates, parameter counts and selected/total update counts. `validation_candidates.csv` and `learning_curves.csv` retain all 135 trials and their epoch histories.
- `dataset_checks.json`: 27 MAT files, 54 distinct symbol/noise seeds, exact Q8.24 feature/target reconstruction, split masks, causal alignment and MATLAB/Python BER agreement. `dataset_sha256.json` identifies the local files; per-condition manifests/export configuration retain physical settings and drive scaling.
- `arithmetic_checks.json`: original 4-12-2 sigmoid engine matches the supplied package's full 256-word RAM after every phase for 64 samples; resized/tanh execution and LMS linear-system convergence pass. Original sigmoid ROM and fixed-point arithmetic source are unchanged. `training_source.json` identifies the ZIP and source changes.
- `sampling_checks.csv`: all three noiseless 40/80 GSa/s short-record checks pass the 0.001 normalized-RMS threshold. MATLAB Code Analyzer passes the three new export/experiment files; Python syntax checks pass. All 135 fits report zero fixed-point saturation events.

MAT datasets, candidate checkpoints and final int32 model/prediction files remain in ignored local run directories. `run_record.json` records their location and the workflow code revision. Compact CSVs/figures and provenance are versioned here.

## Limits and next step

This is an offline, eight-pass Q8.24 adaptation study on a partially calibrated physical model. It does not establish unique neural necessity, superiority over Volterra, measured-device performance, live FPGA operation or hardware power. Nominal BER intervals assume independent bits; correlated errors and training-seed variability require further study. New widths/tanh still need RTL qualification.

Freeze the present settings before independent confirmation with additional symbol/noise seeds and ANN initializations. Extend high-power compressed-driver testing until BER estimates have useful error counts. Volterra comparison, 64-QAM and controlled-rate live adaptation remain subsequent research steps.
