# Reference: LMS FFE versus ANN

One physical A-RoF topology, 16-QAM, 400 mVpp, 0.4 Gbaud, 15 km fiber, 1 m wireless. Component assumptions and selection method are in [the protocol](../../../ADAPTIVE_COMPARISON.md). Each point uses 65,536 held-out symbols (262,144 bits). The primary ANN is selected by validation EVM, not test BER.

| ROP (dBm) | Unequalized BER | LMS BER | Primary ANN BER | LMS EVM (%) | ANN EVM (%) | BER reduction vs LMS |
|---:|---:|---:|---:|---:|---:|---:|
| -22 | 0.0142365 | 0.0139008 | 0.0125847 | 20.540 | 19.736 | 9.5% |
| -20 | 0.00276184 | 0.00204086 | 0.0012207 | 14.723 | 14.548 | 40.2% |
| -17.5 | 0.000534058 | 0.000278473 | 0.000160217 | 11.116 | 11.035 | 42.5% |

Selected primary receivers:

| ROP (dBm) | Receiver | Symbol taps | Hidden units | Activation | Learning rate | Selected epoch | Updates at selected epoch | Parameters |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| -22 | LMS_FFE | 2 | 0 | linear | 0.00025 | 1 | 8192 | 10 |
| -22 | ANN_tanh_16 | 2 | 16 | tanh | 0.004 | 8 | 65536 | 114 |
| -20 | LMS_FFE | 2 | 0 | linear | 0.001 | 1 | 8192 | 10 |
| -20 | ANN_tanh_16 | 2 | 16 | tanh | 0.004 | 8 | 65536 | 114 |
| -17.5 | LMS_FFE | 9 | 0 | linear | 0.00025 | 2 | 16384 | 38 |
| -17.5 | ANN_tanh_16 | 2 | 16 | tanh | 0.004 | 8 | 65536 | 114 |

All candidates receive eight chronological training passes (65,536 updates); validation selects an earlier checkpoint if appropriate. The tables retain all selected activation groups in `selected_parameters.csv` and `test_metrics.csv`, including 12-unit sigmoid versus 12-unit tanh. `validation_candidates.csv` records every tap/rate/architecture trial and saturation count. `learning_curves.csv` records epoch-wise validation metrics. Quantized learning rates appear in the parameter CSV.

[BER plot](ber_comparison.png) · [BER PDF](ber_comparison.pdf) · [EVM plot](evm_percent_comparison.png) · [EVM PDF](evm_percent_comparison.pdf)

These are single-test-sequence, single-initialization conditional simulation results. Nominal BER intervals in the CSV assume independent bits. They do not establish multiseed robustness, superiority over Volterra, uniquely necessary neural equalization, live adaptation throughput, or measured hardware power. New tanh/width variants are not RTL-qualified. The physical model retains the earlier effective gain/noise calibration and its partial-match limitations. Full MAT datasets and Q8.24 state/prediction files remain in ignored local run directories; export configuration, manifests and hashes preserve provenance.
