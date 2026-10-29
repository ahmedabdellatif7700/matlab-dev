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

Figure 13(c) is the priority unequalized comparison: fixed 16-QAM, 400 mVpp, received power -22 to -17.5 dBm. `photodetect` optionally normalizes the two incident optical bands to `received_optical_power_dbm`. The total average power is the sum of their individual band powers; the optical-frequency interference term averages out on a power meter. Both fields receive the same scale factor, preserving their relative powers and modulation. The reference plane is assumed to be the PD input, after fiber and amplification. Transmitter/MZM waveforms do not change across the sweep.

The power setting defaults to NaN, preserving the earlier normalized-field path. `pd_responsivity_a_w=1` is an effective reference value, not a measured detector parameter. Unreported RF amplification and receiver noise remain lumped into one aggregate real post-detector white-noise PSD. This optional power scaling is not a calibration of launched laser powers, an EDFA noise figure or an absolute electrical receiver budget.

`arof.received_power.run` fits effective driver gain and noise using only the -22 and -17.5 dBm endpoints. `arof.received_power.sweep` freezes that fit and generates full noisy waveforms at all ten powers with fresh seeds. Eight intermediate power points are excluded from numerical fitting. The study profile uses 16,384 calibration payload symbols and five independent validation seeds with 65,536 payload symbols each. The conditional Gaussian calibration surrogate is checked by the actual waveform/noise runs. Read-off allowances and source provenance are in `reference_data/README.md`.

This is a separate Figure 13(c) calibration: the Figures 9/11 operating-point fit is not silently repurposed or overwritten. Fixed aggregate receiver noise gives noise-to-signal power proportional to ROP^-2 because beat current is proportional to optical watts. Whether this accounts for the paper's curve is a model test. Shot noise, ASE beat noise, RIN, saturation and other unmeasured components are not claimed to be identified by two fitted endpoints.

`main.m` exposes the received-power comparison as stage 13, controlled by `received_power_comparison_enabled`. To run only this experiment, call `arof.received_power.run(arof.configuration())` after adding `matlab` to the path. Figures, per-seed counts and frozen calibration are saved under `datasets/physical_link/received_power/`; curated results are in `results/received_power/`.

## Minimal implementation

The active package has seven files:

| File | Responsibility |
|---|---|
| `configuration.m` | Physical parameters, output paths, run profile and optional report switch. |
| `transmitter.m` | QAM symbols, calibration prefix, transmit RRC, IF and driver. |
| `optical_link.m` | Single-arm MZM, optical-field propagation, PD power/beat and wireless hop. |
| `receiver.m` | Envelope detection, IF downconversion, matched RRC, prefix-only recovery and BER/EVM. |
| `received_power.m` | Figure 13(c) settings, endpoint calibration, frozen sweep and plots. |
| `simulate.m` | Compose the same physical functions for repeated simulations. |
| `baseline_report.m` | Optional physical/numerical controls and diagnostic plots. |

Related stage functions are grouped as static methods so `main.m` can show each physical stage explicitly without one file per function. There is no object state or additional signal-processing layer. Configuration is independent of the historical `mmw` scenario suite. Set `cfg.profile="smoke"` for a short end-to-end run; the default `"study"` retains the established symbol counts and seeds. `cfg.baseline_report_enabled=false` skips the optional diagnostic report by default.

The previous implementation and voltage-matching experiments are retained under `archive/+arof_history/`. They are excluded from the normal path and are not run by `main.m`. See [archive instructions](archive/README.md). The recorded Figures 9/11 and Figure 13(c) fits remain alternative effective parameterizations, not one jointly validated device model. This refactor preserves the physical equations, noise, calibration targets and result data.
