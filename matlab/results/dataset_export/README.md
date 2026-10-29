# Dataset export: completed study run

Branch: `codex/dataset-export`, based on physical-link commit `a2b9c6f`. Run: `equalizer_datasets_01`.

Nine MAT files were generated at -22, -20 and -17.5 dBm, each with separate training (32,768 symbols), validation (16,384) and test (65,536) files. Total: 344,064 exported rows, each with 10 causal I/Q inputs and 2 continuous I/Q targets. All nine symbol seeds and all nine noise seeds are distinct. No Q8.24 input or target saturation occurred.

The MAT files occupy about 34 MB under `datasets/physical_link/equalizer_datasets/equalizer_datasets_01/`, outside Git. This directory preserves the compact manifest, complete settings and validation record. File SHA-256 hashes in `validation_report.json` identify the generated MAT files; regenerating MAT files may change their header/hash even when numerical arrays are identical.

Python/SciPy loaded every file and checked input/target shapes, causal tap alignment, prefix/training masks, independent seeds, 400 mVpp drive, finite Q8.24 representations and zero saturation. Every input/target word was compared against the actual `to_q` implementation from the supplied `nn_training_model.zip`; all matched. A separate nine-file smoke export passed the same checks.

`X` and `Y` are float64 arrays of exact decoded Q8.24 values. `X_q8_24` and `Y_q8_24` are the signed int32 words. Unnormalized physical I/Q, boundaries and the exact per-file configuration are included. No equalizer training or performance claim is part of this export.

[Dataset schema and commands](../../DATASETS.md). The next step is conventional FFE/Volterra tuning on the training/validation files, preserving final-test independence.
