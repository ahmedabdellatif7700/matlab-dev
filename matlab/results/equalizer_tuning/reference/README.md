# Tuned reference equalizers

Physical conditions are unchanged. Training/validation records match the original experiment; final testing uses fresh seeds. See [the tuning protocol](../../../EQUALIZER_TUNING.md). All neural training/inference uses the supplied Q8.24 stages and continuous I/Q outputs.

| ROP (dBm) | Unequalized BER | Tuned LMS BER | Tuned ANN BER | LMS EVM (%) | ANN EVM (%) |
|---:|---:|---:|---:|---:|---:|
| -22 | 0.0146942 | 0.0144119 | 0.0126572 | 20.508 | 19.390 |
| -20 | 0.00265503 | 0.00206757 | 0.000995636 | 14.784 | 14.147 |
| -17.5 | 0.000770569 | 0.000453949 | 9.15527e-05 | 11.218 | 10.615 |

Primary models selected by validation EVM:

| ROP (dBm) | Receiver | Symbol taps | Hidden units | Activation | Initial learning rate | Selected epoch | Parameters |
|---:|---|---:|---:|---|---:|---:|---:|
| -22 | LMS_FFE | 2 | 0 | linear | 6.25e-05 | 3 | 10 |
| -22 | ANN_tanh_16 | 2 | 16 | tanh | 0.008 | 32 | 114 |
| -20 | LMS_FFE | 2 | 0 | linear | 0.001 | 1 | 10 |
| -20 | ANN_tanh_16 | 2 | 16 | tanh | 0.008 | 32 | 114 |
| -17.5 | LMS_FFE | 9 | 0 | linear | 6.25e-05 | 6 | 38 |
| -17.5 | ANN_tanh_16 | 2 | 16 | tanh | 0.008 | 32 | 114 |

Every fit allows 32 passes (262,144 updates), using learning-rate multipliers 1 / 0.5 / 0.25 at epochs 1 / 17 / 25. Each test point has 65,536 symbols (262,144 bits). All activation/width groups are retained in the CSVs; only validation determines the primary model. Selected epochs and total/selected update counts are explicit in `selected_parameters.csv`.

[Validation convergence](validation_convergence.png) · [BER curve](ber_comparison.png) · [EVM curve](evm_percent_comparison.png). Constellations show identical held-out symbols, common axes and full-test BER/EVM:

- -22 dBm: [PNG](constellations/reference_-22_dbm.png), [PDF](constellations/reference_-22_dbm.pdf).
- -20 dBm: [PNG](constellations/reference_-20_dbm.png), [PDF](constellations/reference_-20_dbm.pdf).
- -17.5 dBm: [PNG](constellations/reference_-17.5_dbm.png), [PDF](constellations/reference_-17.5_dbm.pdf).

This is an offline parameter refinement, not live on-chip adaptation. One initialization and one fresh test sequence per point remain insufficient for a multiseed generalization claim. New input histories/activation configurations need RTL verification; the 256-word capacity refers to model working state, not storage for an entire training record. Low error counts need longer testing. A gain over LMS alone does not establish superiority over Volterra or unique neural necessity.
