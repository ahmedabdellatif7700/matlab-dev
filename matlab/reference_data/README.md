## Active single case

`main.m` fits `single_case_reference.csv`: 16-QAM, 400 mVpp before the driver, approximate EVM 13.2% from Figure 9 and unequalized BER 10^(-3.09) from Figure 11. Both values are calibration targets, not independent validation measurements. Their common received-power condition is an assumption. Simulation validation uses new random seeds after fitting is frozen.

It then compares a frozen-parameter 380–500 mVpp sweep against `single_case_curve_reference.csv`. These are approximate visual readings of the same 16-QAM curves, with the same read-off allowances. Only 400 mVpp enters the fit. The six other voltages test extrapolation; no gain/noise refitting is performed. Five paired seeds and 65,536 payload symbols per seed give 1,310,720 bits per voltage in the study profile. The existing 400 mVpp validation samples are reused. `single_case_curve_enabled` controls this sweep independently of the point fit.

To regenerate only this sweep from the saved fit, run from the repository root:

```matlab
addpath('matlab');
cfg = arof.configuration();
folder = arof.sweep_single_case('matlab/results/single_unequalized_case', cfg, 380:20:500);
```

This writes a new timestamped run under `datasets/physical_link/single_case_curve/`. The curated plots and CSV results are in `matlab/results/single_case_curve/`.

The broader readings below remain exploratory reference material; they are not the active scope.

# Approximate unequalized reference points

Source: Liu et al., JLT 2017, DOI 10.1109/JLT.2017.2717778, Figures 9, 11 and 12 (PDF pages 7 and 8; journal pages 4412–4413).

`unequalized_points.csv` contains approximate manual visual readings of the supplied PDF, not the authors' numerical data. Figure 9 values are EVM in percent; Figures 11/12 values are BER converted from visually read log10(BER). Assigned read-off uncertainty is ±0.5 EVM percentage points or ±0.06 in log10(BER). These are digitization allowances, not experimental confidence intervals. Additional systematic read-off error is possible.

Only 4-QAM EVM at 100 and 500 mV is used to identify effective noise and compressed-driver parameters. The paper's separate approximately 200 mV → 5 Vpp drive statement constrains driver gain. Every other recorded EVM/BER point is excluded from parameter fitting. Figure 11/12 received-power/noise settings are not established to be identical to Figure 9; cross-figure predictions therefore rely on an explicit shared-noise assumption.

Points excluded from numerical parameter fitting have still been inspected during exploratory model development. Their labels do not imply blind validation or independent model selection. New simulation seeds assess Monte Carlo variability, not independent experimental validation.

Figure 13 provides separate received-power sweeps, including different ranges for the 16-QAM drive settings. It does not specify which received-power conditions apply to every point in Figures 9, 11 and 12. A shared-noise comparison is a modeling assumption, not a reported experimental fact.
