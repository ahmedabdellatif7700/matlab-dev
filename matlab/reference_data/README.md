# Approximate unequalized reference points

Source: Liu et al., JLT 2017, DOI 10.1109/JLT.2017.2717778, Figures 9, 11 and 12 (PDF pages 7 and 8; journal pages 4412–4413).

`unequalized_points.csv` contains approximate manual visual readings of the supplied PDF, not the authors' numerical data. Figure 9 values are EVM in percent; Figures 11/12 values are BER converted from visually read log10(BER). Assigned read-off uncertainty is ±0.5 EVM percentage points or ±0.06 in log10(BER). These are digitization allowances, not experimental confidence intervals. Additional systematic read-off error is possible.

Only 4-QAM EVM at 100 and 500 mV is used to identify effective noise and compressed-driver parameters. The paper's separate approximately 200 mV → 5 Vpp drive statement constrains driver gain. Every other recorded EVM/BER point is excluded from parameter fitting. Figure 11/12 received-power/noise settings are not established to be identical to Figure 9; cross-figure predictions therefore rely on an explicit shared-noise assumption.

Points excluded from numerical parameter fitting have still been inspected during exploratory model development. Their labels do not imply blind validation or independent model selection. New simulation seeds assess Monte Carlo variability, not independent experimental validation.

Figure 13 provides separate received-power sweeps, including different ranges for the 16-QAM drive settings. It does not specify which received-power conditions apply to every point in Figures 9, 11 and 12. A shared-noise comparison is a modeling assumption, not a reported experimental fact.
