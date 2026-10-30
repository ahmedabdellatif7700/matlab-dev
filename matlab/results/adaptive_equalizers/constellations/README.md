# Held-out I/Q constellations

Each figure compares **unequalized, LMS FFE and the validation-selected ANN** at the same physical operating point. Black crosses mark ideal unnormalized 16-QAM positions. All panels show the same first 8,192 chronological held-out symbols on equal I/Q axes and shared limits, without extra gain/phase fitting or selective outlier removal. BER/EVM annotations use all 65,536 test symbols, not just the displayed subset.

These plots reuse the saved Q8.24 inference outputs; no training or model selection was repeated. Before plotting, Python checks that predictions reproduce the recorded full-test error counts and EVM. Only the known constellation normalization is undone for display. EVM remains a numerical error metric; the constellation is its complementary I/Q visualization.

| Condition | -22 dBm | -20 dBm | -17.5 dBm |
|---|---|---|---|
| reference | [PNG](reference_-22_dbm.png) / [PDF](reference_-22_dbm.pdf) | [PNG](reference_-20_dbm.png) / [PDF](reference_-20_dbm.pdf) | [PNG](reference_-17.5_dbm.png) / [PDF](reference_-17.5_dbm.pdf) |
| bias offset | [PNG](bias_offset_-22_dbm.png) / [PDF](bias_offset_-22_dbm.pdf) | [PNG](bias_offset_-20_dbm.png) / [PDF](bias_offset_-20_dbm.pdf) | [PNG](bias_offset_-17.5_dbm.png) / [PDF](bias_offset_-17.5_dbm.pdf) |
| driver compression | [PNG](driver_compression_-22_dbm.png) / [PDF](driver_compression_-22_dbm.pdf) | [PNG](driver_compression_-20_dbm.png) / [PDF](driver_compression_-20_dbm.pdf) | [PNG](driver_compression_-17.5_dbm.png) / [PDF](driver_compression_-17.5_dbm.pdf) |

Regenerate from the repository root after MATLAB export and completed Python training:

```sh
python python/equalizer.py --workflow adaptive --run datasets/physical_link/adaptive_equalizers/adaptive_40g_01 --phase report
```

Generated constellations are written under `<run>/python_results/constellations/`. Single-condition training jobs also produce them in their own result directory. `manifest.csv` records the selected ANN, display/test counts and filenames. See [the full comparison](../README.md) for finite-error-count, model-calibration and single-seed limitations. In particular, the high-power compression point has only two ANN errors versus four LMS errors and does not establish a reliable BER advantage.
