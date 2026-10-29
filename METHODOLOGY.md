# Single-link A-RoF equalization methodology

## Agreed scope

Develop one modular 60 GHz analog radio-over-fiber link based on the optical-beating, multilevel-QAM experiment in Liu et al., JLT 2017, DOI [10.1109/JLT.2017.2717778](https://doi.org/10.1109/JLT.2017.2717778). Establish its physical baseline before drawing conclusions about equalizers. Start with 16-QAM; 64-QAM is a later configuration of the same link.

The communication objective is a supported BER or continuous-output EVM advantage for a candidate NN over well-tuned FFE and Volterra equalizers. There is no matched-multiplication-budget requirement. There are no four scenario families in the new study. Earlier bitwise experiments remain historical results, not evidence validating this new physical model.

The physical parameters, assumptions and missing specifications are in [PHYSICAL_LINK.md](matlab/PHYSICAL_LINK.md).

## Implementation sequence and simple commits

| Step | Work | Commit |
|---|---|---|
| 1 | Establish the physical specification; distinguish paper values, modeling assumptions and unavailable measurements. | `01: define physical link` |
| 2 | Build modular MATLAB waveform, IF, optical modulation, fiber, photodetection, wireless/receiver and recovery stages. | `02: build physical link` |
| 3 | Establish low-drive controls, spectra, alignment, BER/EVM and stage checks without equalization. | `03: establish baseline` |
| 4 | Export independent training, validation and test datasets from MATLAB with continuous I/Q targets and explicit boundaries. | `04: export datasets` |
| 5 | Tune and compare FFE and Volterra baselines. | `05: compare conventional equalizers` |
| 6 | Develop the supplied `nn_training_model/` package for the selected NN configurations. | `06: integrate NN training` |
| 7 | Compare activations with matched architectures, initialization, data and training budgets. | `07: compare activations` |
| 8 | Freeze choices and report independent confirmation, figures and the 64-QAM extension. | `08: report comparison` |

Review the physical baseline after step 3 before proceeding to dataset export and receiver-performance claims. Development and pushes remain on `codex/development`; `main` is not modified.

## MATLAB and Python responsibilities

MATLAB owns physical signal generation, channel modeling and dataset export. Python consumes MAT arrays `X` and `Y` for training, validation, testing and inference. `Y` contains transmitted I/Q, not bit logits. BER uses QAM slicing and Gray-bit demapping; EVM uses continuous I/Q estimates. Normalization, timing and gain calibration must be explicit and shared across receiver comparisons.

Use FFE, Volterra and NN only in the new comparison. Tune each receiver using training and validation data, without imposing equal tap counts or multiplication counts. Preserve final test independence. Report useful operating points supported by the physical link, rather than increasing arbitrary impairment parameters until an NN wins.

## Authoritative NN development base

Use the user-supplied `nn_training_model.zip` directory as the development base, with the accompanying user-supplied methodology. Do not substitute a separately written integer engine. Import and extend that directory in step 6, preserving its stages:

1. Load the sample and known target.
2. Forward propagation.
3. MSE loss/output delta.
4. Backward propagation using pre-update weights.
5. Immediate sample-by-sample SGD update.

All stored NN values use Q8.24: samples, targets, parameters, activations, gradients, deltas, learning rate and outputs. Preserve the reference rounding and saturation boundaries, including one rounding per accumulated dot product. Full-width products and wide accumulators are intermediate arithmetic, not mixed stored precision. Do not use the supplied package's optional Q8.8 publishing path in the new workflow.

Network dimensions, memory and activations may be extended, with matching forward/derivative implementations. Establish exact agreement with the supplied reference's existing golden outputs before qualifying extensions. Floating-point computations may serve as diagnostic references only; they do not replace Q8.24 candidate training/inference results.

## Later hardware phase

Target KV260 at the 100 MHz reference point, initially at a controlled sample rate. The existing sequential engine is approximately 85 k training samples/s and 285 k inference samples/s; these rates do not automatically apply to extended networks. Overload drops complete samples and increments counters. Full-rate processing of the communication waveform is a later architecture project.

Live supervised adaptation uses initial known training, payload inference and periodic training bursts with current weights. Register-controlled mode switching exists; stream-aligned training requires flags carried with frames. Training can stop after exactly N accepted training samples. Time-varying-channel adaptation experiments and RTL/power comparisons are outside the immediate physical-baseline checkpoint. Later report both tool-estimated and measured board power where feasible, with measurement boundaries stated.

The superseded methodology is retained in [docs/HISTORICAL_METHODOLOGY.md](docs/HISTORICAL_METHODOLOGY.md) for provenance. Its four-family, AdamW/bitwise and multiplication-budget choices do not govern this work.
