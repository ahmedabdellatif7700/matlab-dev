# Millimeter-wave impairment scenarios

`main.m` runs Scenario 1: a 60 GHz, 1 Gbaud, single-carrier 16-QAM
complex-envelope link with an isolated nonlinear RF driver. `legacy_main.m`
preserves the previous MATLAB/Python/ONNX/Q8.8 workflow. Scenario 1 writes
separate artifacts and does not regenerate those hardware files.

The [validated Scenario 1 report](results/scenario1/README.md) includes the
first full study's curves, metrics and limitations. In this initial study the
NN improves on FFE, but fifth-order Volterra outperforms the NN on pooled
EVM. Most NN fits reach their epoch budget, so these are budget-specific
results, not proof of an intrinsic ranking between model families.

Run from the project root:

```sh
matlab -batch "run('matlab/main.m')"
```

For a short end-to-end smoke check without opening windows:

```sh
matlab -batch "addpath('matlab'); c=mmw.scenario_config(); c.profile='smoke'; mmw.run_pa_benchmark(c);"
```

MATLAB requires Communications Toolbox (`qammod`, `qamdemod`, `rcosdesign`,
`de2bi`). Python requires the existing NumPy, SciPy and PyTorch dependencies.
The interpreter is selected from `MMW_PYTHON`, then the existing
`~/miniconda3/envs/machine-learning/bin/python`, otherwise `python3`.
The runner requires the current branch to be `codex/development`; it never
switches branches, commits, or merges.

## Channel and experiment

The sampled signal is the complex envelope at eight samples per symbol, not
an explicitly sampled 60 GHz sinusoid. Transmit and matched receive RRC
filters have unit energy, rolloff 0.25, and span 24 symbols. The original
12-symbol span has approximately 1.28% residual RMS EVM from truncation at
this rolloff; the longer span reduces it to approximately 0.056%. Known filter delay
is removed analytically and burst edges are discarded. There is no timing or
carrier-estimation error in this first scenario.

After pulse shaping, a fixed-saturation amplifier applies

```
u = |g*x| / A_sat
gain = (1 + u^(2*p))^(-1/(2*p))
phase = phase_limit * u^2 / (1 + u^2)
y = g*x * gain * exp(j*phase)
g = 10^(-IBO_dB/20)
A_sat^2 = E_symbol / samples_per_symbol
```

The default smoothness is 2 and the phase limit is 20 degrees. These are
illustrative Rapp AM/AM and bounded AM/PM characteristics, not measured
device parameters or a calibrated reproduction of a paper's apparatus.
Negative backoff means nominal input power above the saturation reference.

Receiver noise is complex white Gaussian noise added before the matched
filter, with fixed variance `E_symbol / 10^(reference_SNR_dB/10)`.
Reference SNR denotes the ideal linear link at **zero-dB backoff**. At
backoff B, ideal-link SNR is `reference_SNR_dB - B`. Thus increasing backoff
reduces compression but also reduces received SNR. The known transmit gain
`g` is removed for all receivers; noise is amplified by the same factor.
No receiver noise power is recomputed from distorted received power.

Chromatic dispersion, MZM/photodetector nonlinearity, optical noise, wireless
multipath, phase noise, and multiuser interference are excluded. They require
separate scenarios. In particular, a nonlinear RF driver must not be presented
as a physical MZM model, and this scenario's SNR axis is not optical dBm.

The study profile uses:

- Backoff: -3, 0, 3, 6, 9, 12 dB.
- Reference SNR: 24, 30, 36 dB.
- Independent replicate seeds: 42, 43, 44.
- Separate bursts of 12,000 training, 3,000 validation, and 30,000 test
  symbols per condition. Windowing removes one row from each burst.
- Two additional controls: linear/noiseless and linear/30 dB AWGN.

Every partition has its own random stream and pulse filters. For a given
seed, symbols and unit-noise draws are held constant across operating points.
There are no shared waveform samples across training, validation and test.

## Equalizers and fairness

All receivers use the same causal two-complex-symbol window, ordered
`[I[n], Q[n], I[n-1], Q[n-1]]`, and estimate the current transmitted symbol.
No method adapts to test targets or receives extra input history.

| Receiver | Fit and selection |
|---|---|
| Before EQ | Current received sample, after known transmit-gain removal |
| Complex FFE | Two complex taps, training-only least squares, no intercept |
| Linear I/Q | Full real linear map and intercept, validation-selected ridge |
| Volterra 2/3/5 | All real I/Q monomials through the specified order, including cross-lag terms; validation-selected ridge |
| NN 4-12-2 | Sigmoid hidden layer, linear output; Adam and best validation checkpoint |

Volterra orders 2, 3 and 5 have respectively 30, 70 and 252 real fitted
coefficients, including two intercepts. The NN has 86. Polynomial input and
feature normalization, and NN input normalization, use training data only.
Polynomial fitting uses FP64 SVD; the NN uses deterministic FP32 CPU training
with one thread. Research-only Adam settings are 0.003 learning rate, batch
size 512, up to 250 epochs, and patience 35. They do not alter `config.py` or
the existing SGD/Q8.8 workflow. Each condition starts a fresh NN.

Ridge candidates minimize training mean squared residual plus lambda times
squared coefficients. Selection minimizes validation MSE. The unpenalized
intercept is handled by centering. The model includes all degrees up to its
order, so cubic terms are not omitted merely because order two was used in
one paper. No trained receiver is forced to perform worse than the NN.

## Results and interpretation

Each invocation creates a new directory under
`datasets/scenarios/pa_single_carrier/<timestamp>/`. An existing run ID is
refused. The directory contains configuration JSON/MAT, source snapshots,
revision/dirty status, per-condition datasets, fitted models, predictions,
validation histories, metrics, dataset hashes, software versions, and logs.
`COMPLETE` is written only after all conditions and plots finish.

`all_metrics.csv` retains each replicate. `pooled_metrics.csv` sums bit errors
and bit counts, and computes pooled reference-normalized EVM from error and
reference energies. EVM uses no fitted test gain or phase. BER uses MATLAB's
exported constellation labels, rather than assuming Python's QAM ordering.

Nominal 95% two-sided Clopper-Pearson intervals are reported. Bits within a
nonlinear channel can have correlated errors, so these binomial intervals
are approximate evidence; replicate results must also be examined. Zero
errors are not proof of zero BER. On logarithmic plots they appear as downward
triangles at the upper confidence bound, without connecting them as measured
BER values. The default reference `3.8e-3` is a literature comparison, not an
implemented FEC code or a universal decoder threshold. `target_status`
compares the nominal interval with this reference.

Figures are exported as PNG, vector PDF and editable MATLAB FIG:

- RMS EVM versus backoff, with one panel per reference SNR.
- BER versus backoff and BER versus reference SNR.
- Matched constellation panels for the same test symbols.
- Training/validation NN convergence at a representative operating point.
- EVM versus direct-evaluation multiplication count.

Complexity counts include direct polynomial feature products and real dot
products, using four real multiplications per complex multiplication. They
exclude affine preprocessing, which can be folded into coefficients, and
the implementation cost of sigmoid. The NN additionally needs 12 sigmoid
evaluations. These counts are not FPGA area, throughput or power estimates.

The smoke profile uses one seed, two backoffs, one reference SNR, shorter
bursts, and 30 NN epochs. It validates execution, not convergence or low BER.
This scenario may show FFE/Volterra performing as well as or better than the
NN. Such an outcome is a result, not an implementation failure. Higher memory,
other NN architectures and additional physical impairments are future studies.

## Relation to the supplied papers

- Liu et al., JLT 2017, DOI `10.1109/JLT.2017.2717778`: motivates
  compression/drive sweeps and constellation, EVM, BER and convergence figures.
  Its MZM/heterodyne link and complex multilevel network are not implemented
  by this isolated RF-driver scenario.
- Liu et al., PTL 2017, DOI `10.1109/LPT.2017.2753174`: adaptive-alpha
  convergence study uses BPSK and separate alpha/weight learning rates.
  Adaptive alpha is deferred.
- Liu et al., OFC 2018, DOI `10.1364/OFC.2018.W4B.3`: joint SC-FDMA
  multiuser equalization is deferred.
- Song et al., JLT 2024, DOI `10.1109/JLT.2024.3416046`: motivates
  performance/complexity comparisons. Its SSB optical link, 64-QAM and OFDM
  are distinct from this scenario. Its reference regression NN uses 22 real
  inputs, whereas this study deliberately starts with the existing four.

## Validation

```sh
python -B -m unittest discover -s python/tests -v
matlab -batch "addpath('matlab'); r=runtests('matlab/test_pa_scenario.m'); assertSuccess(r);"
```

The numerical tests check joint-I/Q polynomial recovery and serialization,
bit-count BER and zero-error confidence bounds, pulse-filter alignment,
MATLAB bit mapping, independent partitions, repeatability, and the expected
6 dB SNR loss when increasing backoff by 6 dB at fixed receiver noise. A
noiseless strong-drive comparison at 8 and 16 samples per symbol checks
oversampling convergence to within 0.5% RMS.
