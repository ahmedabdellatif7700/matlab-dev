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

## Unequalized experimental-curve comparison

The earlier exploratory helper `arof.replicate_unequalized` covers multiple published curves; it is no longer the active main path. `main.m` now calls `arof.match_unequalized_case` for only 16-QAM at 400 mVpp. This remains step 3, before any equalizer comparison. The reference points are approximate manual readings of Figures 9, 11 and 12; provenance and read-off allowances are in `reference_data/README.md`.

The comparison fixes physical topology and tests unresolved device assumptions explicitly:

- Linear IF driver, amplitude-envelope RF detector.
- Bandpass-envelope Rapp driver, amplitude-envelope RF detector.
- The same compressed driver with a square-law RF detector.

Only the approximately 200 mVpp → 5 Vpp voltage anchor and Figure 9 4-QAM EVM at 100/500 mV enter numerical calibration. Small-signal gain, saturation and aggregate IF-noise PSD are inferred parameters; they are not reported device measurements. Rapp smoothness is fixed at 2. The simulation never adds arbitrary AM/PM or echoes.

The real post-detector noise PSD is fixed across drive and modulation and scales correctly with simulation sample rate. It represents unresolved aggregate receiver noise in normalized detector-output units, not independently identified ASE, thermal, shot or RIN contributions. Each detector hypothesis needs a noise calibration in its own output units. Per-waveform transmitter peak scaling establishes the pre-driver Vpp axis; normalization depends on the finite waveform's PAPR.

The remaining reference points are excluded from numerical fitting, but they have been inspected during exploratory modeling. They are not blind validation. Figures 9, 11 and 12 do not establish identical optical received-power/noise settings, so a shared-noise cross-figure comparison is conditional. The paper's complete EVM normalization and RF detector response are unavailable.

Report overlays, raw errors/counts, nominal BER intervals, EVM seed spread and numerical discrepancies. A fitted point or qualitative trend match is not successful quantitative replication. Model qualification and equalizer work remain pending this comparison and review. `paper_comparison_enabled` and `paper_profile` in `+arof/configuration.m` control this experiment; the noiseless diagnostic configuration remains unchanged.

Figure 13 also varies received power separately, with different plotted power ranges for different drive settings. It does not establish a common operating power for Figures 9, 11 and 12. Therefore mismatches in a shared-noise cross-figure prediction cannot uniquely identify a faulty component or its missing transfer curve.

## Current single operating point

The active targets are approximately 13.2% continuous-output EVM (Figure 9) and BER 10^(-3.09) (Figure 11), both at 16-QAM and 400 mVpp before the driver. Both metrics identify an effective operating-point driver gain and lumped real IF-noise PSD. Gain is constrained to 12.5–25, corresponding approximately to 5–10 Vpp at the MZM, rather than extrapolating an unreported full driver curve. This gain is not claimed to reproduce amplifier behavior at other inputs.

A Gaussian decision-cell surrogate is used only for smooth parameter identification, using noiseless recovered sample means and prefix-only receiver gain. The complete physical model then injects actual noise with five preselected fresh seeds. Calibration and validation symbol counts match. Parameter fitting never uses those validation errors. Report raw bit counts, nominal intervals and EVM read-off tolerance; do not select favorable seeds or silently change targets.

A successful result establishes a conditional calibrated single-point match. It does not establish that Figures 9 and 11 used identical received optical power, uniquely identify the missing component parameters, or independently reproduce the full experiment. Earlier 16-QAM/200 mVpp results matched EVM only and do not provide a published same-point BER target.

## Frozen-parameter voltage curve

`arof.sweep_single_case` extends the 400 mVpp calibration to 380–500 mVpp in 20 mV steps. Driver gain, noise PSD and all component settings remain fixed. It performs complete physical-link simulations at the other voltages, reusing the five preselected seeds and the cached 400 mVpp validation counts. `main.m` shows this explicitly after the single-point fit; `single_case_curve_enabled` and `single_case_curve_mv` control the sweep. A saved fit can be swept directly without repeating calibration.

`arof.plot_single_case_curve` exports EVM/BER overlays as PNG and vector PDF, with approximate Figure 9/11 readings and their read-off allowances. EVM bars show seed standard deviation; nominal BER intervals assume independent bits. Within each seed, waveform/noise draws are paired across voltages. These bars are not an independent experimental-validation claim. A mismatch away from 400 mVpp identifies a limitation of extrapolating this effective gain/noise model, rather than grounds for silently refitting each point.
