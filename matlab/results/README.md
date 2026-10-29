The active [single unequalized case](single_unequalized_case/README.md) is 16-QAM at 400 mVpp. Both published metric targets were used in calibration; fresh-seed simulations give 13.193% EVM and BER 8.255e-4. This is a calibrated point match, not independent reconstruction of the experimental hardware or a result from an equalizer.

The current single-link physical baseline is [here](physical_baseline/README.md). It establishes unequalized signal recovery and numerical consistency, not equalizer or NN performance. The studies below use earlier models and remain historical results.

# Validated millimeter-wave equalizer studies

Five completed studies contain **306 seeded operating-point runs** and **4050 receiver/user metric rows**. Each condition uses three independent replicate seeds; operating points share random draws within a seed for paired comparisons. Multiuser aggregate rows are not additional independent experiments.

| Study and full figures | Runs | Metric rows |
|---|---:|---:|
| [1: RF driver](scenario1/README.md) | 60 | 420 |
| [2: MZM / adaptive activation](scenario2/README.md) | 69 | 576 |
| [3: DSB / SSB optical fading](scenario3/README.md) | 72 | 576 |
| [4: Four-user SC-FDMA](scenario4/README.md) | 39 | 1950 |
| [5: 28 GHz OFDM](scenario5/README.md) | 66 | 528 |

Each report contains held-out BER/EVM, constellation figures, numerical CSVs and source/dataset provenance. Large waveforms, saved models, vector PDFs and editable MATLAB FIGs remain in the ignored per-run dataset directories.

## Where is an NN necessary?

[operating_regions.csv](operating_regions.csv) screens every tested condition and user. A non-neural receiver counts as meeting the 3.8e-3 literature reference only when its nominal 95% upper BER bound lies below it. An NN-only candidate requires a neural upper bound below the reference and **every evaluated non-neural lower bound above it**, including the untrained front end. Overlapping bounds remain inconclusive.

This screen found **0 NN-only candidates** across the listed condition/user combinations. This is exploratory evidence within the stated models, observation windows and training/complexity budgets, not universal NN necessity. The bounds are nominal binomial intervals: correlated errors and scanning many operating points limit statistical interpretation. Confirm a candidate with additional independent data before claiming a reproducible operational advantage.

## Interpretation

- The RF-driver study establishes a useful non-neural baseline; fifth-order Volterra is competitive.
- MZM drive sweeps distinguish noise-limited, nonlinear and severely folded regimes. Multilevel output can lower EVM by pulling estimates toward discrete QAM levels, so BER must be examined alongside EVM.
- DSB fading near a null can limit every receiver. SSB avoids that magnitude fading; it is not an NN effect.
- The multiuser study gives all joint model families the same cross-user observations and also evaluates individual receivers. Joint and individual NN parameter budgets differ and are reported, rather than interpreted as a pure information-only experiment.
- OFDM compares pilot estimates with a known-linear-shape diagnostic. Global nonlinear ICI is not generally recoverable from only three neighboring carriers. Failure of this small post-equalizer is not a limitation of all neural OFDM receivers.

See [equations, controls, caveats and reproduction commands](../SCENARIOS.md). No experimental paper curve or optical-power calibration is claimed reproduced.

## Four-case flexible bitwise equalizer study

The newer [four-case study](design_study/README.md) uses variable neural architectures and validation-selected activations, with 12 independent confirmation runs plus frozen-receiver SNR curves. All four cases meet the locked BER and multiplication-count criterion. This concerns resource-constrained performance, not NN-only necessity across unrestricted conventional receivers; it does not change the historical five-scenario screen above. Conditions, neural tuning parameters, baseline choices and full BER/EVM tables are included.
