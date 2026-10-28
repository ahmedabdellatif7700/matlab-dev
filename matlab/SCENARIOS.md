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

## Scenario 4: joint multiuser SC-FDMA equalization

Four independent 16-QAM users use localized DFT-spread OFDM with 16-point
DFT spreading, a 256-point IFFT and 32-sample cyclic prefix. The sample rate
is 4 GHz. Each user occupies 250 MHz; guards of 0/4/12 bins correspond to
0/62.5/187.5 MHz. This is a 60 GHz complex-envelope uplink aggregation study,
not an exact replication of the paper's 66.7 Mbit/s/user waveform.

Each branch has an illustrative RF PA (3/6/9/12 dB backoff by user), known
branch drive-gain normalization before aggregation, and user-specific linear
RF memory shorter than the cyclic prefix. A shared quadrature MZM processes
the *sum* of the branch waveforms at drive indices 0.3/0.7/1.1/1.5. Its
nonlinearity creates cross-user terms that cannot be modeled by adding
independently distorted users afterward. Fiber loss is normalized; optical
dispersion, Kerr effects and phase noise are excluded in this scenario.

Receiver FFTs and subcarrier de-mapping are followed by a per-bin linear
channel estimate fitted **only to the training partition's known payload**.
That estimate is frozen for validation and test, then followed by inverse
DFT spreading. This is supervised front-end calibration; its training cost
is shared by every post-equalizer. No validation/test payload labels enter
the channel estimate. Noise is fixed at 28 dB active-subcarrier Es/N0 at
unit MZM drive; it is not normalized to received signal power.

Every post-equalizer sees current/previous symbols within each DFT-spread
block; windows never cross block boundaries. Joint methods observe both taps
from all four users (16 real inputs, eight real outputs); individual methods
observe only their own two complex taps. Individual and joint FFE, Volterra,
and NN receivers are compared, so access to other users is not confused with
an exclusively neural benefit. Joint multilevel NN has 32 hidden real units;
each individual multilevel NN has 12. Hidden and output activations use the
same four-level function as Scenario 2.

The joint full cubic polynomial has 968 nonconstant features. Full joint
fifth order would require 20,348, exceeding the predeclared 1,000-feature
budget; it is explicitly listed as *not evaluated* in manifests, never counted
as a failed receiver. Individual cubic and fifth-order models are evaluated.
All computation/parameter counts are totals for recovering all users, not
silently divided per user. Per-user and aggregate BER are saved separately.

Independent bursts contain 256/64/512 SC-FDMA blocks for training/validation/
test; each contributes 15 supervised symbols per user after windowing.
Three replicate seeds are used. Tests verify nonoverlapping allocations,
guard spacing, the noiseless roundtrip, unity training channel estimate,
within-block windows and repeatability. Spectra show shared-link regrowth;
guard bands can reduce adjacent leakage but cannot eliminate all inter-user
cross-modulation.

```sh
matlab -batch "addpath('matlab'); c=mmw.suite_config(4); mmw.run_suite(c);"
matlab -batch "addpath('matlab'); r=runtests('matlab/test_multiuser_scenario.m'); assertSuccess(r);"
```

Reference: Liu et al., OFC 2018, DOI `10.1364/OFC.2018.W4B.3`. The paper
motivates joint versus individual processing; it does not justify giving
only the neural receiver access to the interfering users.

## Scenario 5: 28 GHz wideband OFDM and channel-estimation limits

Single-band 16/64-QAM OFDM occupies 52 active carriers (bins -26:-1 and
1:26) in a 64-point base FFT with a 16-sample base cyclic prefix. Fourfold
oversampling uses a 256-point IFFT and 64-sample CP. Subcarrier spacing is
1 GHz/52, so the sample rate is approximately 4.923 GHz. The occupied
bandwidth is 1 GHz, rather than calling the sampling rate the bandwidth.

The RF PA is the same illustrative Rapp/AM-PM model, now swept at -3, 0,
3, 6 and 9 dB backoff. A fixed linear echo channel has delays of two and four
base samples, with coefficients -0.35*exp(j*0.4) and 0.15*exp(-j*0.6), both
shorter than the cyclic prefix. An ideal small-signal SSB optical link spans
10 km at 1550 nm; known bulk phase/delay is removed as in Scenario 3. There
is no DSB fading imposed on an SSB link. Fiber attenuation and optical noise
remain normalized/excluded; this is not calibrated optical-power sensitivity.

Each independently generated train/validation/test burst contains eight
known random pilot OFDM symbols before its payload, plus unused burst-edge
guard symbols. Pilot generation has a random stream independent of payload
generation. A test verifies that changing payload symbols cannot change the
noiseless pilot channel estimate. LS estimates are obtained only from those
known pilots, never from validation or test payload labels.

Two front ends are compared under otherwise identical waveforms:

- `LS`: the per-subcarrier estimate from eight noisy pilot symbols.
- `known`: a diagnostic with the known linear channel *shape* and a common
  complex gain fitted from the same pilots. This removes per-bin estimation
  error without giving the receiver payload labels. It is not an oracle for
  nonlinear distortion and is not assumed deployable with an unknown channel.

Noise variance is Es/10^(28/10) in the unitary-FFT convention, so reference
SNR is active-subcarrier Es/N0 at zero backoff. Its level is fixed as drive
changes; ideal linear-link Es/N0 becomes 28-backoff dB. Gain removal, pilots
and channel corrections are shared by every post-equalizer at an operating
point. Pilot shape-NMSE removes a common gain for diagnosis only; raw NMSE
including nonlinear compression bias is also saved.

Post-equalizers observe current, previous and next *frequency* carrier within
one OFDM payload symbol. Edge carriers without neighbors are excluded. No
feature window crosses an OFDM-symbol boundary. All methods therefore use
six real observations, with 24 real hidden NN units or the same full
Volterra feature budgets as Scenario 3. Nonlinear ICI can depend on the whole
OFDM block: a three-carrier feed-forward NN need not succeed, and failure
does not establish a limitation of all neural OFDM receivers.

At the study burst sizes, there are 115/34/288 payload OFDM symbols for
training/validation/test and 50 supervised carrier positions per symbol.
Pilots, cyclic prefix, inactive carriers and excluded edge positions are
not counted as payload test bits. No claim of a paper's bit rate is made.

Additional figures compare pilot shape mismatch with backoff and transmitter
PAPR for OFDM versus RRC single-carrier signals of the same occupied
bandwidth over approximately equal-duration 52 ns intervals. PAPR is measured
before the PA. The single-carrier curves are transmitter diagnostics, not BER
curves from a mismatched channel. Tests check noiseless 16/64-QAM recovery,
the cyclic prefix/known channel, payload-independent pilots and convergence
between fourfold and eightfold oversampling under strong nonlinear drive.

```sh
matlab -batch "addpath('matlab'); c=mmw.suite_config(5); mmw.run_suite(c);"
matlab -batch "addpath('matlab'); r=runtests('matlab/test_ofdm_scenario.m'); assertSuccess(r);"
```

Reference: Song et al., JLT 2024, DOI `10.1109/JLT.2024.3416046`. The paper's
NN improvements and OFDM 64-QAM limitations motivate this comparison, but
the experimental device responses and received optical power are not known.

## Running and reviewing the complete suite

```sh
matlab -batch "addpath('matlab'); run_all_scenarios('study','off');"
```

For a short end-to-end check replace `study` with `smoke`. Smoke settings
are not convergence or low-BER evidence. New run IDs prevent overwriting
previous results. Receivers execute from frozen source copies inside each
run directory, so later scenario development cannot change a running fit.
Compact full-study figures, CSVs and provenance are retained in
`matlab/results/scenario2` through `scenario5`; large datasets/checkpoints and
editable/vector figures remain in the ignored per-run dataset directories.

These studies use illustrative models and explicitly bounded observation,
model-order and training budgets. Report BER alongside EVM: multilevel output
can pull estimates toward discrete constellation points and sharply lower
EVM even when decision errors change little. An NN-only operational region
must be demonstrated against tuned baselines at a specified target/budget;
it is never assumed from the presence of nonlinearity alone.
