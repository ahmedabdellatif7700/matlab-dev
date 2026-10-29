# Archived A-RoF experiments

`+arof_history/` preserves the implementation before the minimal-module refactor, including the single-point EVM/BER fit, voltage curves and earlier broad paper-matching attempts. It is not on the normal MATLAB path. Function calls use the separate `arof_history` namespace to avoid conflicts with the active `arof` modules.

The active link is in `../+arof/`; `../main.m` shows its physical stages and runs the received-power comparison. Existing plots, CSV files and calibration records remain in `../results/`.

To rerun an archived voltage sweep from a saved fit, start from the repository root:

```matlab
addpath('matlab');
addpath('matlab/archive');
cfg = arof_history.configuration();
folder = arof_history.sweep_single_case('matlab/results/single_unequalized_case', cfg, 380:20:500);
```

The archive retains its original calibration flags and numerical behavior. Its old report commands refer to the archived namespace. The separate `historical_main.m`, `legacy_main.m` and `+mmw/` belong to earlier research and hardware workflows and are not dependencies of the active physical model.
