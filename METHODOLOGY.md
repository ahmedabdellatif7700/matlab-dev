# Methodology: neural equalization of a 60 GHz link

## 1. What we are studying

We are studying whether a compact neural receiver can compensate nonlinear millimeter-wave link impairments more effectively than the tested conventional receivers within a limited inference budget.

The practical question is:

> At a specified channel condition and BER target, how much receiver complexity is needed, and does a neural receiver provide a useful advantage?

We compare BER, hard-decision EVM, parameter count, explicit real multiplications and activation evaluations. We retain higher-cost conventional receivers in the comparison, even when they match or outperform the NN.

The present results support a **bounded arithmetic/performance advantage**. They do not establish that these impairments can only be solved by neural networks, or that fewer multiplications automatically mean lower hardware power or latency.

## 2. Why these impairments and cases

The supplied Liu and Song papers motivate studying modulator nonlinearity, RF amplifier distortion, channel memory and optical dispersion. We use controlled simulation models to separate these effects and study their combinations. The model equations and controls are documented further in [SCENARIOS.md](matlab/SCENARIOS.md).

We are not reproducing the papers' experimental apparatus, optical-power calibration or measured curves. Our cases use a 60 GHz complex-envelope carrier and single-carrier QAM.

| Case | Conditions selected for confirmation | Why we study it |
|---|---|---|
| 1 | 16-QAM; MZM drive 1.5; reference SNR 30 dB | Test nonlinear amplitude distortion without adding an explicit echo or optical dispersion stage. |
| 2 | 64-QAM; MZM drive 1.05; one-symbol echo of magnitude 0.2 and phase 36°; reference SNR 40 dB | Test nonlinear distortion together with channel memory and more closely spaced constellation levels. |
| 3 | 16-QAM; PA input backoff −3 dB; AM/PM limit 40°; one-symbol echo of magnitude 0.25 and phase 45°; reference SNR 32 dB | Test simultaneous amplitude compression, amplitude-dependent phase rotation and memory. |
| 4 | 64-QAM; PA input backoff 3 dB; AM/PM limit 40°; 0.3 km small-signal DSB fiber model; reference SNR 44 dB | Test RF nonlinearity together with frequency-dependent optical transport. |

MZM drive is a dimensionless modulation-depth setting in the implemented RF-fundamental model. The AM/PM setting is an asymptotic phase limit, not a constant rotation applied to every sample. Reference SNR is defined relative to the ideal, undistorted link; it is not the measured SNR after every impairment.

## 3. Step 1 — Generate the transmitted waveform in MATLAB

Generate independent random symbol sequences and map them to Gray-coded 16-QAM or 64-QAM. Use:

- Symbol rate: 1 Gbaud.
- Waveform sampling: eight samples per symbol.
- Transmit and receive root-raised-cosine filters: rolloff 0.25, span 24 symbols.
- A 60 GHz carrier represented through its complex envelope and its role in the optical response; we do not sample the RF carrier directly.

Pulse shaping comes before the nonlinear stage. This matters because even a discrete constellation with a small number of amplitude levels produces a continuously varying waveform envelope after pulse shaping.

The source is currently random QAM, not a coded transmission with an implemented FEC decoder.

## 4. Step 2 — Apply the impairment stages

The implemented signal flow is:

```text
Random symbols → Gray QAM → transmit RRC
    → MZM fundamental OR RF PA
    → optional optical DSB response
    → optional one-symbol echo
    → receiver AWGN
    → receive RRC → known delay/gain removal → symbol sampling
```

**MZM:** use the fundamental RF-band response of an ideal quadrature-biased intensity modulator. Its envelope scaling contains the Bessel-function factor `2 J1(m)/m`, with modulation depth `m` proportional to waveform amplitude. The model rejects the RF harmonic bands and uses a narrowband envelope approximation. It is not a full large-signal optical-field simulation.

**RF PA:** use Rapp amplitude compression and a bounded amplitude-dependent phase rotation. Lower input backoff drives the PA more strongly. Applying this model to the pulse-shaped waveform allows nonlinear distortion to interact with the transmit and receive filters.

**Optical DSB transport:** use the small-signal carrier/sideband-beating dispersion response. This produces frequency-dependent RF fading. The PA/optical cascade is a phenomenological model, not a calibrated transmitter model. The large-signal MZM approximation is not cascaded with this small-signal optical approximation in the selected cases.

**Echo:** add a delayed, scaled complex copy of the waveform at a delay of one symbol. This is a controlled memory impairment, not a measured multipath channel.

**Noise:** add complex Gaussian receiver noise with variance referenced to the undistorted symbol energy. Do not renormalize noise to the distorted signal power; that would conceal attenuation or compression penalties.

At the receiver, remove the known filter delay and configured drive gain, and trim burst edges. Timing, carrier recovery and channel-parameter estimation are therefore idealized. Their acquisition costs and errors are outside this study.

## 5. Step 3 — Give each receiver the same observations

Each receiver uses the current and previous received complex samples:

`x[n] = [I[n], Q[n], I[n−1], Q[n−1]]`.

This gives four real inputs. All receivers have the same observation window; the NN does not receive extra neighboring symbols or hidden channel information.

The target symbol and its Gray-bit label come from the transmitted sequence. Any input standardization is fitted using training data only. Validation and test data use those fitted statistics.

The conclusions apply to this two-complex-sample window. A longer conventional filter, a different nonlinear model or a different receiver structure may change the comparison.

## 6. Step 4 — Establish conventional receiver controls

| Receiver | Purpose |
|---|---|
| Complex FFE | Establish what a linear receiver can recover. |
| Full real polynomial/Volterra regression, orders 1, 2, 3, 5 and 7 | Test increasingly expressive nonlinear symbol reconstruction, including cross-products between I/Q and temporal inputs. |
| Polynomial bit classifiers, orders 1, 2, 3 and 5 | Control for the benefit of directly optimizing bit decisions rather than reconstructing I/Q by squared error. |
| Gaussian RBF bit classifiers, 16, 64 and 256 centers | Include a non-neural nonlinear decision-boundary model outside the polynomial family. |

Regression receivers reconstruct I/Q and use Cartesian constellation slicing. Bit classifiers predict bit decisions. Ridge regularization and kernel bandwidth are selected using validation data. Baseline parameter tables report optimizer convergence where applicable.

These controls matter because an improvement over FFE alone would not establish an advantage over nonlinear conventional receivers. Likewise, a bitwise NN should not be compared only against squared-error regression without bitwise controls.

## 7. Step 5 — Tune neural architecture and activation in Python

Use the shared `RealEqualizer` in [equalizer.py](python/equalizer.py), with choices centralized in [config.py](python/config.py). Hidden width and depth are configurable; the research receiver is not restricted to the legacy 4–12–2 topology.

The main search includes shallow and deeper networks, tanh, SiLU, softsign, fixed smooth multilevel and learned smooth multilevel activations. A focused 64-QAM extension also tests smaller networks and narrow bottlenecks.

The learned multilevel activation adjusts shared transition locations and positive slopes. It is an experimental activation candidate inspired by multilevel responses. Selecting it for a case does not, by itself, establish mathematical novelty or isolate the activation's contribution from all other tuning choices.

The network outputs one logit per Gray bit: four outputs for 16-QAM and six for 64-QAM. A logit at or above zero gives bit 1. We optimize bitwise binary cross-entropy because BER is the primary performance objective.

For the four selected networks, training uses AdamW with learning rate 0.003, weight decay 1e−4 and minibatches of 512, for up to 600 epochs. Validation patience is 70 epochs, followed by 12 LBFGS polishing steps. Retain the checkpoint with the lowest validation BCE. The focused discovery extension uses a separate 900-epoch budget; it is recorded in the search history.

| Case | Real network dimensions | Selected activation | Trainable parameters | Real multiplications/symbol | Activation evaluations/symbol |
|---|---|---|---:|---:|---:|
| 1 | 4 → 4 → 4 | tanh | 40 | 32 | 4 |
| 2 | 4 → 16 → 6 | Learned multilevel | 188 | 224 | 48 |
| 3 | 4 → 16 → 4 | tanh | 148 | 128 | 16 |
| 4 | 4 → 16 → 16 → 6 | tanh | 454 | 416 | 32 |

The final dimensions denote bit logits, not reconstructed I/Q components.

## 8. Step 6 — Separate discovery from confirmation

| Stage | Data and decisions | Reason |
|---|---|---|
| Discovery | Seed 501; 10,000 training and 10,000 validation symbols per candidate; no test payloads generated | Explore conditions, architectures, activations and conventional hyperparameters without inspecting final tests. |
| Design lock | Save selected conditions, architectures and BER targets in `selection.json` | Prevent the final tests from determining the design. |
| Confirmation | Seeds 1201, 1202 and 1203; each uses 18,000 training, 6,000 validation and 50,000 test symbols | Assess the locked choices on fresh data and independent realizations. |

These are generated symbol counts. Forming a two-sample window removes the first symbol from each exported segment.

The discovery study began with 12 conditions and expanded to 18 using higher-SNR and moderate-backoff controls. The target-selection rule was also refined during discovery. These decisions were made before confirmation data existed; they were not all fixed before the initial discovery sweep. The [protocol history](matlab/results/design_study/protocol_history.md) records this distinction.

For each confirmation seed, retrain the locked NN architecture from three initializations and select the lowest validation-BCE checkpoint. Conventional hyperparameters are also selected using that seed's training/validation data. Test BER never selects a restart or hyperparameter.

If confirmation fails, report the failure. A revised design requires another discovery/confirmation round with fresh final test data.

## 9. Step 7 — Define the performance/complexity criterion

During discovery, require NN validation BER no greater than `3.8e−3` and a validation BER gap of at least `1e−4` relative to the best tested classical receiver below 1.5 times the NN multiplication count.

A BER target may come from the configured reference grid or from the midpoint between those validation BERs. Therefore, some targets are **validation-derived design targets**, not standardized communication requirements. Their origin is recorded and the targets are frozen before confirmation.

A confirmed case must satisfy all three conditions on pooled confirmation data:

1. The NN meets its locked BER target.
2. Every tested classical receiver with no more multiplications than the NN fails that target.
3. The cheapest tested classical receiver meeting the target requires at least 1.5 times the NN's explicit real multiplications.

The `3.8e−3` level is a literature reference, not a demonstrated post-FEC guarantee. No FEC decoder is implemented.

Polynomial arithmetic counts assume reuse of lower-order monomials. Counts exclude additions and fusible affine normalization. Activation evaluations are reported separately because a sigmoid, tanh or division has a different cost from a multiplication.

These are inference arithmetic comparisons. We have not measured implementation latency, throughput, FPGA resources, energy or power for these new networks. Classical inference uses float64 and neural inference float32; parameter counts must not be interpreted as equal-precision memory comparisons.

## 10. Step 8 — Evaluate BER, EVM and uncertainty

**BER** is the number of wrong Gray bits divided by the number of evaluated bits. Report raw errors and bit counts alongside BER.

**Hard-decision EVM** compares the decided constellation symbols with the transmitted symbols:

`EVM (%) = 100 × sqrt(mean(|s_decided − s_transmitted|²) / mean(|s_transmitted|²))`.

This is a common decision-level metric for the compared receivers. It is not continuous equalizer-output EVM and does not establish that the analog waveform has been reconstructed.

Report each seed separately and pool error counts across the three seeds. Include nominal Wilson BER intervals and paired contiguous-symbol-block bootstrap differences between each receiver and the NN. The bootstrap uses 128-symbol-scale blocks and 2,000 resamples per comparison.

Nominal binomial intervals assume independent bits; symbol errors and temporal dependence can violate that assumption. Seed variation and paired block comparisons provide additional evidence. The intervals are not a multiple-comparison-corrected proof of universal superiority.

## 11. Step 9 — Plot SNR generalization without retuning

After training, freeze every receiver. Generate fresh payloads at reference-SNR offsets of −6, −3, 0, +3 and +6 dB from each selected condition.

Evaluate the same fitted receivers at every point. Select the best-classical comparison using validation BER for each seed, not the SNR-sweep test results. Record the underlying selected model in the CSV.

Within a seed, SNR points reuse the transmitted symbols and scale the same noise realization. This supports paired curve comparisons; the points are not independent replications. Sweep payloads are separate from the confirmation payloads.

These curves measure robustness to SNR mismatch. They must not be described as receivers retrained or optimized independently at every SNR.

## 12. What the completed study shows

| Case | Pooled NN BER | FFE BER | Best classical BER within the NN multiplication budget | Cheapest classical/NN multiplication ratio at the locked target |
|---|---:|---:|---:|---:|
| 1 | 7.48e−4 | 4.30e−3 | 4.28e−3 | 2.06× |
| 2 | 2.15e−3 | 7.54e−3 | 3.30e−3 | 1.66× |
| 3 | 5.87e−4 | 5.95e−3 | 2.44e−3 | 2.90× |
| 4 | 8.41e−4 | 8.60e−3 | 1.35e−3 | 2.36× |

All four cases meet the locked criterion. Higher-cost classical receivers sometimes achieve lower BER than the NN. The defensible conclusion is that the selected NNs offer useful BER at a smaller explicit multiplication budget than the tested classical matches under these conditions.

The [complete report](matlab/results/design_study/README.md) contains the figures, full receiver comparisons, physical conditions, network settings, fitted activation coefficients and baseline hyperparameters.

## 13. Implementation responsibilities and next research steps

| Location | Responsibility |
|---|---|
| [matlab/main.m](matlab/main.m) | Run the complete staged experiment. |
| [matlab/+mmw/design_config.m](matlab/+mmw/design_config.m) | Physical link settings, payload counts and seeds. |
| [matlab/+mmw/design_candidates.m](matlab/+mmw/design_candidates.m) | Candidate impairment conditions. |
| [python/config.py](python/config.py) | Network/activation candidates, learning settings and baseline search choices. |
| [python/equalizer.py](python/equalizer.py) | Shared neural model, training/inference and Python entry point. |
| [python/activations/](python/activations) | Activation implementations. |
| [python/tuning/](python/tuning) | Discovery selection, confirmation and reporting. |

MATLAB and Python exchange MAT files using `X`, `Y` and explicit split counts. Research inference returns decided symbols and bit predictions. The original no-argument Python workflow and its legacy ONNX/fixed-point interface remain available.

The new research results are floating point. The existing Q8.8 inference and Q8.24 training work do not establish fixed-point performance for the new architectures or activations. Quantization-aware evaluation and hardware support are subsequent steps.

For a stronger research contribution, the next steps are to isolate the activation effect with matched architectures and training budgets, broaden conventional receiver/window choices, calibrate channel parameters against measured or published apparatus, and assess equal-precision implementation costs. These are future experiments, not conclusions already demonstrated by the present four cases.
