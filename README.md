The priority paper comparison is **unequalized 16-QAM BER versus received optical power at 400 mVpp**, using Figure 13(c). [Received-power results and assumptions](matlab/results/received_power/README.md). Two endpoint readings calibrate effective gain and receiver noise; eight intermediate powers test the frozen model. This is a separate effective fit from the earlier [single EVM/BER point](matlab/results/single_unequalized_case/README.md) and [voltage curves](matlab/results/single_case_curve/README.md), not a jointly validated apparatus reconstruction. No equalizer is used. `main.m` retains the separate low-drive physical diagnostic; its near-zero BER is not the impaired case.

To run only the received-power comparison from the repository root:

```matlab
addpath('matlab');
cfg = arof.configuration();
folder = arof.match_received_power(cfg);
```

# Current work: one physical A-RoF link

Run `matlab/main.m` to establish the modular physical baseline described in [METHODOLOGY.md](METHODOLOGY.md) and [matlab/PHYSICAL_LINK.md](matlab/PHYSICAL_LINK.md). This stage runs no equalizer. The agreed review checkpoint is the physical baseline before dataset export and comparison. See [baseline results and assumptions](matlab/results/physical_baseline/README.md).

NN development will use the supplied `nn_training_model/` package, preserving its training stages and Q8.24 arithmetic; the separate draft integer engine is not used. Import/development of that package is step 6 of the agreed sequence.

The previous campaign entry point is preserved as `matlab/historical_main.m`. Documentation and results below describe historical work and do not qualify the new physical model.

---

# Millimeter-wave neural equalizer studies

MATLAB waveform/channel models and matched FFE, Volterra and neural-network
receiver comparisons. Development is on `codex/development`; the runners
refuse other branches and never commit, switch branches or merge themselves.

| Scenario | Experiment |
|---|---|
| 1 | 60 GHz single-carrier RF amplifier compression and AM/PM |
| 2 | MZM drive, 16/64-QAM, memory, multilevel and adaptive-alpha activations |
| 3 | Combined RF distortion and DSB/SSB optical dispersion fading |
| 4 | Four-user SC-FDMA with individual versus joint receivers |
| 5 | 28 GHz wideband OFDM, pilot channel estimation and nonlinear ICI |

Select a number in [matlab/main.m](matlab/main.m), or run every study:

```sh
matlab -batch "addpath('matlab'); run_all_scenarios('study','off');"
```

Replace `study` with `smoke` for a short execution check. MATLAB needs the
Communications and Signal Processing toolboxes; Python dependencies are in
`python/requirements.txt`. Set `MMW_PYTHON` to select the Python interpreter.

[Physical equations, assumptions and scenario commands](matlab/SCENARIOS.md)
explain observation/training budgets, independent data partitions, fixed
noise references, pilot use and numerical checks. [Results](matlab/results/README.md)
contain EVM/BER curves, matched constellations, diagnostics and provenance.
NN superiority is tested rather than assumed; severe information loss can
limit every receiver. A literature BER line does not imply FEC decoding.

The original ONNX/Q8.8 hardware-oriented workflow remains available in
`matlab/legacy_main.m`; the research suite does not overwrite its artifacts.

```sh
python -B -m unittest discover -s python/tests -v
matlab -batch "addpath('matlab'); run_numerical_tests();"
```

The staged 60 GHz design study is now the `matlab/main.m` entry point: physical link → discovery-only tuning → locked four-family selection → fresh confirmation → BER/resource tables. Read [the design protocol](matlab/DESIGN_STUDY.md) before interpreting NN efficiency claims. The existing five-scenario study and its historical results remain available.

The completed [four-case bitwise equalizer study](matlab/results/design_study/README.md) includes physical-condition and NN-parameter tables, three-seed held-out results, and BER/SNR figures. It finds lower multiplication counts than the tested classical receivers at locked BER targets; activation costs and higher-cost classical matches remain visible.

For a step-by-step explanation of what we study, why each stage is needed and how the comparisons are controlled, read [METHODOLOGY.md](METHODOLOGY.md).
