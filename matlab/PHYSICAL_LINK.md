# Physical link specification

## Architecture and source

Source: Liu et al., *A Multilevel Artificial Neural Network Nonlinear Equalizer for Millimeter-Wave Mobile Fronthaul Systems*, JLT 35(20), 2017, DOI [10.1109/JLT.2017.2717778](https://doi.org/10.1109/JLT.2017.2717778), especially experimental setup/Fig. 8 and drive discussion/Fig. 9.

```text
Gray QAM → RRC → 0.8 GHz electrical IF → driver → single-arm MZM
  + second optical tone → optical coupler → SMF → EDFA → photodetector
  → mmWave amplification/1 m wireless hop → envelope detector
  → IF/baseband recovery → matched RRC → timing/gain calibration
```

The MZM is driven at IF. The two optical tones beat near 60 GHz at the photodetector. This is different from directly driving a 60 GHz MZM or applying the usual directly modulated RF-DSB dispersion-fading formula.

## Reported parameters

| Item | Value in the paper | Implementation interpretation |
|---|---|---|
| Modulation | Single-carrier 16-QAM and 64-QAM | Start with 16-QAM. |
| Symbol rate | 0.4 Gbaud | Preserve for the initial baseline. |
| RRC rolloff | 0.75 | Occupied baseband width is 0.7 GHz. |
| Electrical IF | 0.8 GHz | Positive IF band approximately 0.45–1.15 GHz. |
| AWG | 4 GSa/s | Initial waveform sample rate; 10 samples/symbol. |
| Laser wavelengths | 1552.90 and 1552.42 nm | Use their actual calculated frequency separation, approximately 60 GHz. |
| Modulated laser | 1552.90 nm | Single-arm MZM on this optical tone. |
| MZM bandwidth | 10 GHz | Do not mistake this for a 60 GHz modulation bandwidth. |
| Quadrature bias | Vpi/2 = 2.2 V | Vpi = 4.4 V. |
| Fiber | 15 km standard SMF | Optical-field dispersion, not an imposed RF fading curve. |
| EDFA | At remote access unit before photodetection | Gain/noise figure are not available in the extracted setup description. |
| PD bandwidth | 60 GHz | A 3 dB bandwidth is not a brick-wall cutoff or a complete measured response. |
| Wireless | 1 m, horn antennas of 20 dBi | Path gain/phase can be lumped in the normalized diagnostic. |
| Receiver | Envelope detection to IF; 20 GSa/s capture; offline recovery | Derive the correct detection envelope; no coherent-LO receiver substituted. |

The paper reports approximately 200 mV peak-to-peak before the driver corresponding to approximately 5 V peak-to-peak at the MZM. This is a specific experimental point, not a verified full driver compression curve or a universal operating limit. Do not derive arbitrary PA AM/PM settings from it.

## Explicit assumptions for the first diagnostic

- Model a balanced single-arm MZM with field transfer `(1 + exp(j*pi*(Vbias+v)/Vpi))/2`. It retains the phase associated with single-arm modulation. Exact device chirp, extinction ratio and insertion loss are unavailable.
- Represent the two optical bands separately. Propagate each with its offset-dependent dispersion phase and analytically extract their photodetection beat term. Do not sample the optical carrier or place a 60 GHz tone on a 4 GSa/s grid.
- Use nominal SMF dispersion 17 ps/(nm km) and attenuation 0.2 dB/km as declared engineering assumptions, not measurements from this experiment. They remain configurable and must be revisited for calibrated reproduction.
- Start with normalized optical amplitudes, unit lumped RF-amplifier gains and an ideal Friis hop using the reported distance/antenna gains, ideal timing and no stochastic noise. There is no claim of calibrated optical watts, received optical dBm, electrical SNR or BER sensitivity at this stage.
- Use a declared one-pole 10 GHz modulator response and one-pole 60 GHz PD response as provisional bandwidth approximations. These are not measured component transfer functions.
- Use low electrical drive, specified in RMS volts, as a linear-limit diagnostic. Report actual waveform Vpp as well. It is not claimed to reproduce the paper's nonlinear optimum near 5 Vpp.
- Reconstruct the 4 GSa/s AWG waveform on an internal 20 GSa/s numerical grid before optical modulation. This is an ideal DAC/reconstruction assumption, not the paper's oscilloscope model; sampling convergence must be checked.
- RRC truncation, IF extraction bandwidth and burst guards are numerical design choices; validate their effect separately from physical distortion.
- The RF envelope detector uses an ideal amplitude law `abs(rf_envelope)`. Its measured response and operating regime (including any square-law behavior) are unavailable. This is separate from optical square-law photodetection; it must be revisited before calibrated nonlinearity claims.
- EDFA gain/noise, laser phase/frequency noise, PD saturation and driver compression remain disabled until justified parameters are supplied. Deterministic fiber attenuation may be compensated by an explicitly ideal noiseless gain for diagnostics; this is not an EDFA noise model.

## What establishes the baseline

Check symbol recovery with the nonlinear stage replaced by its local linear field response; optical beating frequency and RF band placement; correct single-arm field/intensity relation; zero-length and finite-length field propagation; amplitude/delay alignment; spectral guard behavior; and sensitivity to numerical filter truncation. Then report the actual nonlinear model's unequalized constellation, BER and continuous-output EVM at the same low drive.

A low-drive, noiseless recovery result establishes implementation consistency only. It does not calibrate the apparatus, validate a practical noise floor or demonstrate a neural advantage. Before physical BER-versus-power claims, resolve or bound component gain/noise, optical powers, PD/RF frequency responses and synchronization assumptions.

## Received optical power sweep

The current link uses fixed 16-QAM at 400 mVpp, with ROP from -22 to -17.5 dBm. `photodetect.m` normalizes both optical bands to the total time-average power at the PD input, preserving their relative powers and modulation. `pd_responsivity_a_w=1` remains an effective reference convention. Fiber and amplification settings remain unchanged across powers.

`link_configuration.m` retains the earlier effective gain 17.097235738210383 and aggregate post-detector IF-noise PSD 1.996608740342124e-25. Their provenance is `results/received_power/frozen_calibration.json`. They are conditional fitted values, not independently measured device parameters. The earlier curve was a partial quantitative match; removing the comparison tools does not change that qualification.

`ber_vs_rop.m` runs only the physical model and plots the simulated BER curve. It does not load paper targets, fit parameters, compute paper-match scores or run diagnostic experiments. The default uses five seeds, 65,536 payload symbols per seed and ten powers. Configure shorter runs through `cfg.sweep_seeds`, `cfg.sweep_payload_symbols` and `cfg.rop_dbm`. Outputs are per-seed counts, pooled BER/EVM, nominal Wilson bounds, run configuration and PNG/PDF curves under `datasets/physical_link/ber_vs_rop/`. Zero-error points are shown at their nominal upper bound. Bit dependence limits the interpretation of nominal intervals.

## Implementation: physical stages only

`main.m` connects these ordinary functions, each in its own file in `+arof/`:

| Stage | File |
|---|---|
| QAM symbols and known prefix | `generate_symbols.m` |
| Transmit RRC pulse shaping | `pulse_shape.m` |
| IF modulation and analog reconstruction | `upconvert.m` |
| Electrical driver | `driver.m` |
| Single-arm MZM and second optical tone | `modulate.m` |
| Fiber propagation and ideal amplification | `propagate.m` |
| Received optical power and photodetection | `photodetect.m` |
| Wireless hop | `wireless.m` |
| Envelope detection and baseband conversion | `downconvert.m` |
| Matched RRC, prefix timing/gain and sampling | `recover.m` |
| Payload BER and EVM | `measure.m` |

The shared FFT-filter implementation is in `+arof/private/filter_envelope.m`. Configuration and the BER-versus-ROP runner are outside the package, in `link_configuration.m` and `ber_vs_rop.m`. There are no active grouped classes, fitting functions, comparison tools, diagnostic reports or transmitter/receiver/simulation wrappers.

The earlier tools and archived implementations have been removed from the working tree; their source remains in Git history. Stored figures and data are preserved. The current cleanup changes organization and removes refitting from normal execution; it does not claim additional experimental validation.
