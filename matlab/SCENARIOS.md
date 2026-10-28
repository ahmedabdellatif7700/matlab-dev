# Paper-informed impairment scenarios

Select a scenario number in `main.m`; the original hardware workflow remains
in `legacy_main.m`. Scenario 1's existing guide is [README.md](README.md).
Research scenarios use isolated per-run directories and never overwrite the
legacy ONNX or Q8.8 artifacts. All work is on `codex/development`.

## Scenario 2: nonlinear MZM and activation comparisons

This is a quadrature-biased, ideal chirp-free intensity modulator with direct
detection and an ideal RF bandpass that rejects DC and RF harmonics. It is
an equivalent RF-over-fiber model, not the two-laser heterodyne apparatus in
Liu's experiment. Its optical intensity transfer is

```
P_out/P_in = (1 + cos(pi/2 + pi*v_RF/Vpi))/2
m = mu * |x|/sqrt(P_ref)
y = mu*x * 2*J1(m)/m
```

The RF fundamental follows the Jacobi-Anger expansion. A fixed detector
gain/sign is removed. `mu` is the reference RMS *envelope* modulation index,
not peak-to-peak drive or real RF RMS; `V_envelope,rms/Vpi = mu/pi`, and
`V_RF,rms/Vpi = mu/(sqrt(2)*pi)`. At vanishing input, `2*J1(m)/m -> 1`.
At large drive it folds and can destroy distinguishability. `mzm_envelope.m`
is numerically tested against direct evaluation of the cosine transfer and
Fourier extraction of its RF fundamental. Narrowband envelope separation is
assumed; full optical nonlinear propagation is not simulated here.

Drive values 0.2, 0.6, 1.0, 1.4 and 1.8 are tested for 16-QAM and 64-QAM,
both without extra memory and with a postmodulator one-symbol echo of
`0.2*exp(j*pi/5)`. This echo is an explicit illustrative linear bandwidth/
memory model, not a fitted component response. Receiver noise has fixed
variance referenced to `mu=1` and 30 dB SNR. Reducing drive therefore lowers
linear-link SNR by `20*log10(mu)` dB. Known drive scaling is removed equally
for all receivers. Optical dispersion, fiber Kerr effects and laser phase
noise are excluded to isolate modulator compression and I/Q coupling.

Independent training/validation/test bursts contain 6,000/1,800/15,000
symbols, with separate filtering and random streams. Seeds 42/43/44 are
independent replicates; signal/noise realizations are shared across operating
points within each partition. All receivers use the same current/previous
two-complex-symbol window. Higher-memory models are a separate study.

FFE and full joint-I/Q Volterra orders 2, 3, and 5 use training-only least
squares or validation-selected ridge. NNs use a joint-real one-hidden-layer
architecture with 24 hidden units, not a claim that complex weights are
necessary. Both sigmoid/linear-output and multilevel hidden/output models
are compared. The latter uses Liu's piecewise multilevel sigmoid (`c1=6`)
with four levels for 16-QAM and eight for 64-QAM. It is an architectural
ablation, not an exact reproduction of the paper's complex 10-10-1 network.

At `mu=1` without extra memory, fixed-alpha=4 and trainable shared-alpha
initially 4 are added to the comparison. BPSK cases with memory at reference
SNRs 12 and 18 dB provide an additional adaptive-activation comparison.
All variants start from identical seeded linear-layer weights, receive the
same training data, and use the same hidden width. Adam uses learning rate
0.003 for weights and 0.001 for trainable alpha, up to 400 epochs, with
validation patience 50. This differs from the paper's plain SGD, and neither
its convergence iteration count nor its BER sensitivity gain is asserted.
Up to 15 full-batch L-BFGS steps (10 internal iterations each) polish training
MSE; only a validation improvement can replace the chosen checkpoint.
Alpha is unconstrained, as in the steepest-descent parameterization.

Run and validate:

```sh
matlab -batch "addpath('matlab'); c=mmw.suite_config(2); mmw.run_suite(c);"
matlab -batch "addpath('matlab'); r=runtests('matlab/test_mzm_scenario.m'); assertSuccess(r);"
python -B -m unittest discover -s python/tests -v
```

The generic suite saves every receiver prediction and independently checks
its Gray-QAM BER and reference-normalized EVM in MATLAB. There is no test-set
gain fitting, test-target adaptation, or fixed-point claim. Confidence bounds
are nominal Clopper-Pearson binomial bounds; channel errors can be correlated.
Compact tracked figures and reports are in `matlab/results/scenarioN/`.

References: Liu et al., JLT 2017, DOI `10.1109/JLT.2017.2717778`; Liu et al.,
PTL 2017, DOI `10.1109/LPT.2017.2753174`. The ideal cosine transfer is also
documented in the [OptiCommPy physical models](https://opticommpy.readthedocs.io/en/latest/models.html).
These references motivate the model and experiment; device parameters are
illustrative and experimental curves are not copied or claimed reproduced.

## Scenario 3: RF nonlinearity plus DSB/SSB dispersion fading

The 60 GHz link now compares small-signal optical double sideband and single
sideband transport, with the electrical RF driver either linear or nonlinear
(fixed 3 dB backoff). The optical modulator is deliberately linearized: the
large-signal MZM model in Scenario 2 is **not** blindly cascaded with a
small-signal optical transfer and presented as full nonlinear optical physics.

For D=17 ps/(nm km), lambda=1550 nm and kappa=pi*D*lambda^2*L/c in SI units,

```
H_DSB(f) = cos(kappa*(fc+f)^2)
H_SSB(f) = exp(j*kappa*f^2)
```

The SSB response removes the known carrier phase and linear group delay from
the single optical sideband/carrier beat, leaving residual dispersive phase.
Its magnitude is unity; DSB has power-fading nulls. The first 60 GHz carrier
null occurs at approximately 1.019 km. Lengths 0, 0.5, 0.9, 1.02, 1.15 and
1.5 km straddle this null. Zero-guarded FFT convolution uses 4096 samples on
each side. Tests verify zero-length identity, unity SSB magnitude, the DSB
null location and guard convergence on a bandlimited burst.

Noise is fixed relative to the uncompressed zero-backoff link (30 dB), and
does not decrease when DSB fading attenuates the signal. Known drive gain is
removed after matched filtering. Fiber attenuation, EDFA noise, chirp, Kerr
nonlinearity and phase noise are excluded to isolate fading versus RF
compression. There is no arbitrary gain correction fitted on test targets.

All receivers have three complex observations ordered current, next, previous,
with one symbol of buffering latency. The six real inputs and 24 hidden NN
units are research architectures, separate from the fixed hardware network.
Full fifth-order Volterra has 461 nonconstant features within the declared
1000-feature budget. Model selection and test isolation match Scenario 2.
Deep DSB fading and resulting noise enhancement can limit every receiver;
linear dispersion alone is not evidence that neural processing is necessary.

```sh
matlab -batch "addpath('matlab'); c=mmw.suite_config(3); mmw.run_suite(c);"
matlab -batch "addpath('matlab'); r=runtests('matlab/test_optical_scenario.m'); assertSuccess(r);"
```

Song et al., JLT 2024, DOI `10.1109/JLT.2024.3416046`, uses SSB to avoid
dispersion fading; its 28 GHz / 10 km experiment is a different operating
point. This comparison demonstrates the mechanism at the project's 60 GHz
carrier rather than copying its optical power or BER curves.
