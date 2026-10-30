# Supplied Q8.24 training model

Development base: user-provided `nn_training_model.zip`, member directory `nn_training_model/hardware/training/`. The imported Python stages are forward, MSE loss, backward with pre-update weights, and immediate sample-by-sample SGD update. No optional Q8.8 publishing path is used.

`fixedpoint.py` and `act_sigmoid.py` retain the supplied source. The atomics/controller reference the configurable memory-map module rather than copied address constants. `memory_map.configure(4,12)` preserves the original addresses and 256-word layout; other widths extend that same datapath. A process has one active global layout, so differently sized models run sequentially.

`act_tanh.py` adds `2*sigmoid(2z)-1` using the supplied ROM, with the smooth-function derivative surrogate `1-h*h`. This extension and resized networks are Python algorithm models, not verified RTL. All stored values remain Q8.24 with the supplied arithmetic boundaries.

Verification compares all 256 RAM words against an independently imported original package after each of four phases for 64 training samples. Extended-width/tanh execution and LMS linear-system convergence are checked separately. Compact verification outcomes accompany the experiment results; no additional verification runner is required to train/infer.
