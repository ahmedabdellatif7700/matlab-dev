# LMS FFE versus ANN: one physical A-RoF link

Run on `codex/development`. `+arof/` remains physical stages only. This comparison uses the same 0.4 Gbaud, single-carrier 16-QAM, 15 km fiber, 1 m wireless, optical-beating link as the established baseline. MATLAB owns waveforms; Python owns equalizer training/inference. It does not fit impairments to make an ANN win.

## Physical conditions

| Condition | MZM bias | Electrical driver | Interpretation |
|---|---:|---|---|
| reference | 2.20 V | linear, gain 17.0972357382 | Existing effective link parameters. |
| bias_offset | 2.42 V | same linear driver | +0.22 V = 0.05 Vpi = 9 degrees bias-phase error. Static during a run; not a time-varying tracking experiment. |
| driver_compression | 2.20 V | envelope Rapp, saturation 3.5 V, smoothness 2 | Smooth IF-driver compression before MZM; same small-signal gain. |

All conditions use 400 mVpp before the driver, Vpi=4.4 V, fixed aggregate noise PSD, and ROP values -22, -20 and -17.5 dBm. There is no added arbitrary AM/PM law, echo or uncalibrated laser-phase impairment. The RF amplitude-envelope receiver is retained.

Bias drift is a real modulator issue; [NIST's bias-control work](https://www.nist.gov/publications/accurate-bias-point-control-electrically-isolated-mach-zehnder-interferometric) describes maintaining quadrature despite temperature/coupling changes. MZM and amplifier nonlinearities are discussed in [Perez Santacruz's A-RoF dissertation, Sec. 2.6](https://pure.tue.nl/ws/portalfiles/portal/218503197/20221028_Perez_Santacruz_hf.pdf). The physical topology and multilevel nonlinear equalization motivation follow [Liu et al., JLT 2017](https://doi.org/10.1109/JLT.2017.2717778). These sources motivate the mechanisms; the 0.22 V offset and 3.5 V saturation are declared engineering sensitivity assumptions, not measured specifications of the paper's equipment. The compression is in the electrical IF driver, not a newly calibrated 60 GHz wireless power amplifier.

The Rapp model is Aout=Ain/[1+(|Ain|/Asat)^(2p)]^(1/(2p)), applied to the analytic IF envelope. At p=2 and Asat=3.5 V, its input-envelope amplitude for 1 dB compression is approximately 3.06 V. Report actual simulated MZM Vpp from export metadata rather than calling Asat a measured voltage or component P1dB.

The comparison uses a **40 GSa/s internal numerical grid**; AWG remains 4 GSa/s. A 20/40 GSa/s short-record check exceeded the pre-existing 0.1% normalized-RMS threshold for the bias-offset case. The same check at 40/80 GSa/s passes for all three conditions. This is numerical refinement, not a change in physical baud rate or impairment strength. Existing standalone baseline/ROP code is unchanged.

## Run

```matlab
addpath('matlab');
folder = compare_adaptive_equalizers('study');
```

Then, from the repository root:

```sh
python python/equalizer.py --workflow adaptive --run /path/printed/by/MATLAB
```

For independent condition jobs, add `--only reference`, `--only bias_offset`, or `--only driver_compression`. Their outputs use separate directories. After all finish, collect plots/tables using the same entry point with `--phase report` and without `--only`. Candidate checkpoints permit interruption/resume with identical settings.

Use `compare_adaptive_equalizers('smoke')` only for execution checks. The study exports 8,192 training, 4,096 validation and 65,536 final-test payload symbols per condition/ROP. Every split has a distinct symbol/noise seed. Prefix timing/gain recovery uses only known prefix symbols. Data normalization is the fixed theoretical constellation RMS; no test-payload statistics are used. Configurable export remains outside `+arof/`.

## Receivers and selection

- LMS FFE: two-output real/widely-linear adaptive FIR, identity initialization, 2/5/9 complex-symbol taps and learning rates 0.00025/0.001/0.004. The ordinary complex-linear FFE is a subset of this I/Q mapping; the benchmark is explicitly wider than a strictly complex-linear LMS filter.
- ANN: 4 inputs (two causal I/Q symbol taps), one hidden layer, two linear I/Q outputs. Candidates are 12 sigmoid units, 12 tanh units and 16 tanh units. Rates are in `python/config.py: ADAPTIVE`.
- Both use MSE, chronological batch-1 updates and eight passes through the same training record. Repeated training passes are an offline adaptation study, not a demonstration of live changing-channel tracking.
- Select tap count/width/activation/rate/checkpoint using **validation EVM only**, then freeze weights for test BER/EVM. Parameter/activation groups are also reported separately, including the equal-width 12-unit activation comparison. A larger tanh network does not by itself isolate an activation benefit.
- Report both total updates spent during fitting and updates represented by the selected epoch. There is no matched multiplication-count requirement. FFE may use more input history than the ANN.

Each operating point currently uses one physical test sequence and one ANN initialization seed. Nominal Wilson BER intervals assume independent bits and do not establish across-seed robustness. Additional independent channel realizations and initialization repeats are needed for publication-strength generalization claims. Test results are not used to pick the reported primary ANN.

## Supplied Python stages and arithmetic

`python/hardware/training/` is imported from the user's `nn_training_model.zip`; it is the development base. Forward → MSE loss → backward with pre-update weights → SGD update is unchanged. The baseline 4-12-2 sigmoid configuration must match the original package's complete 256-word RAM after each phase. Layout configuration preserves those addresses and extends the same atomics for other widths. The tanh activation reuses the supplied sigmoid ROM as 2*sigmoid(2z)-1 with derivative 1-h² in Q8.24. Its derivative is the smooth-function surrogate evaluated at quantized h, consistent with the supplied sigmoid derivative convention, rather than the exact local slope of every ROM segment.

All stored samples, weights, biases, activations, deltas, gradients, learning rates and predictions use signed Q8.24. Products/sums are wide integer intermediates. Import rounding is ties-to-even as in supplied `to_q`; internal shifts are nearest ties-away, with explicit saturation and one rounding per dot-product sum. Floating-point arithmetic is used only for physical I/Q, normalization, initialization definitions and reporting metrics. No float32 conversion, Q8.8 path or substitute training engine is used in this workflow.

New widths/tanh are Python models, not verified RTL implementations. Existing user-reported FPGA throughput does not automatically apply to them. Final model states and predictions are saved as int32 words under the ignored run directory; compact results are curated separately.

A result against LMS FFE does not establish superiority over Volterra or prove neural equalization uniquely necessary. The current question is whether the tested neural receivers improve on this tuned adaptive linear baseline under declared component assumptions.
