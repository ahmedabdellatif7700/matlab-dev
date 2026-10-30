# Driver compression: LMS FFE versus ANN

One physical A-RoF topology, 16-QAM, 400 mVpp, 0.4 Gbaud, 15 km fiber, 1 m wireless. Component assumptions and selection method are in [the protocol](../../../ADAPTIVE_COMPARISON.md). Each point uses 65,536 held-out symbols (262,144 bits). The primary ANN is selected by validation EVM, not test BER.

| ROP (dBm) | Unequalized BER | LMS BER | Primary ANN BER | LMS EVM (%) | ANN EVM (%) | BER reduction vs LMS |
|---:|---:|---:|---:|---:|---:|---:|
| -22 | 0.0122566 | 0.0120125 | 0.011116 | 19.918 | 19.277 | 7.5% |
| -20 | 0.00122833 | 0.000911713 | 0.00062561 | 13.963 | 13.833 | 31.4% |
| -17.5 | 4.57764e-05 | 1.52588e-05 | 7.62939e-06 | 10.077 | 10.194 | 50.0% |

Selected primary receivers:

| ROP (dBm) | Receiver | Symbol taps | Hidden units | Activation | Learning rate | Selected epoch | Updates at selected epoch | Parameters |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| -22 | LMS_FFE | 2 | 0 | linear | 0.00025 | 8 | 65536 | 10 |
| -22 | ANN_tanh_16 | 2 | 16 | tanh | 0.004 | 8 | 65536 | 114 |
| -20 | LMS_FFE | 2 | 0 | linear | 0.00025 | 8 | 65536 | 10 |
| -20 | ANN_tanh_16 | 2 | 16 | tanh | 0.004 | 8 | 65536 | 114 |
| -17.5 | LMS_FFE | 2 | 0 | linear | 0.00025 | 8 | 65536 | 10 |
| -17.5 | ANN_tanh_16 | 2 | 16 | tanh | 0.004 | 8 | 65536 | 114 |

All candidates receive eight chronological training passes (65,536 updates); validation selects an earlier checkpoint if appropriate. The tables retain all selected activation groups in `selected_parameters.csv` and `test_metrics.csv`, including 12-unit sigmoid versus 12-unit tanh. `validation_candidates.csv` records every tap/rate/architecture trial and saturation count. `learning_curves.csv` records epoch-wise validation metrics. Quantized learning rates appear in the parameter CSV.

[BER plot](ber_comparison.png) · [BER PDF](ber_comparison.pdf) · [EVM plot](evm_percent_comparison.png) · [EVM PDF](evm_percent_comparison.pdf)

These are single-test-sequence, single-initialization conditional simulation results. Nominal BER intervals in the CSV assume independent bits. They do not establish multiseed robustness, superiority over Volterra, uniquely necessary neural equalization, live adaptation throughput, or measured hardware power. New tanh/width variants are not RTL-qualified. The physical model retains the earlier effective gain/noise calibration and its partial-match limitations. Full MAT datasets and Q8.24 state/prediction files remain in ignored local run directories; export configuration, manifests and hashes preserve provenance.

At -17.5 dBm, the apparent BER reduction is only two errors versus four; nominal intervals overlap, and LMS EVM is lower. This point does not establish a reliable BER advantage. Compression can soften the downstream MZM distortion; it is not necessarily a more impaired total link than the linear-driver reference.
