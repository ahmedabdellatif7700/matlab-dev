# Equalizer refinement on the fixed physical link

This refinement changes receiver parameters, not optical power points, component impairments, gain, noise PSD or recovery. It preserves the supplied Q8.24 forward → MSE → backward with pre-update weights → batch-1 SGD stages. Continuous I/Q estimates are plotted without snapping outputs to ideal QAM decisions.

## Preset fixed before final testing

`python/config.py: ADAPTIVE_TUNED` defines:

- 32 chronological passes instead of eight through the same 8,192 training symbols.
- Learning-rate multipliers 1 for epochs 1–16, 0.5 for 17–24, 0.25 for 25–32. The engine receives a Q8.24 learning-rate word at each stage; no floating-point weight updates are used.
- LMS FFE: 2/5/9 symbol taps; initial rates 0.0000625, 0.00025, 0.001. This also strengthens the conventional control.
- ANN: two symbol taps with 12/16 sigmoid or tanh hidden units; an additional three-tap, 16-unit tanh observation window. Initial sigmoid rate 0.016; tanh rates 0.004/0.008.
- Eight ANN candidates and nine LMS candidates per operating point. ANN layouts remain at most 6-16-2, with 146 parameters, fitting the Python engine's existing 256-word RAM capacity. Configurable dimensions/activations still need corresponding RTL qualification; fitting RAM is not proof of measured hardware throughput or power.

Validation EVM selects the rate, observation window and epoch checkpoint within each activation/width group, then the primary ANN across groups. All candidate choices and validation curves are retained. No final-test BER/EVM influences tuning. Since histories and rates can differ, this refinement is not a controlled activation-only ablation.

## Independent confirmation and fair before/after comparison

Keep the original training/validation symbol and noise seeds. Export fresh final-test sequences by adding 40,000 to the original test symbol seeds (noise remains symbol seed +900,000). Each test still has 65,536 payload symbols. Confirm original and new train/validation arrays match exactly before training.

```matlab
addpath('matlab');
folder = compare_adaptive_equalizers('study','adaptive_tuned_01',40000);
```

```sh
python python/equalizer.py --workflow adaptive --preset tuned \
  --run datasets/physical_link/adaptive_equalizers/adaptive_tuned_01
```

Independent condition jobs can add `--only reference`, `--only bias_offset` or `--only driver_compression`. After all finish, use `--phase report` to combine figures and constellations. The baseline preset remains unchanged. Candidate checkpoint directories reject incompatible settings.

Evaluate the **previously frozen** baseline LMS and ANN weights on the same fresh test files as the tuned receivers. Comparing only against the previous test's BER would confound tuning with a changed test sequence. Preserve prediction words and raw errors, as well as full-test EVM and identically sampled constellation panels.

The 32 passes remain offline reuse of known symbols; they are not live adaptation measurements. The learning-rate schedule is an experimental host policy, not a newly demonstrated stream-control implementation. One fresh test seed and one ANN initialization per point do not establish multiseed robustness. Extremely small error counts require longer independent confirmation. A gain over LMS does not prove superiority over Volterra or unique neural necessity. Report an absent or small improvement honestly.

For the paired before/after figures, collect the condition jobs with:

```sh
python python/equalizer.py --workflow adaptive \
  --run datasets/physical_link/adaptive_equalizers/adaptive_tuned_01 \
  --phase report \
  --baseline-run datasets/physical_link/adaptive_equalizers/adaptive_40g_01
```

The `before_after/` subdirectory records old frozen and newly tuned receivers on the same fresh test symbols.
