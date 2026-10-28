# Joint-user study findings

Joint processing is not automatically superior under the selected two-symbol
observations and training/complexity budgets. Individual receivers are often
competitive. At guard 12 and common drive 1.5, aggregate BER is approximately
0.01766 for the joint multilevel NN versus 0.01911 for the individual NNs,
but both remain above the 3.8e-3 reference. Other operating points favor the
individual models. Per-user curves must be checked alongside aggregates.

The joint NN has 808 real parameters; the four individual NNs have 344 in
total. Therefore this is not a controlled information-only experiment with
identical NN capacity. Parameter and operation counts are retained. Joint
full cubic Volterra, individual cubic/fifth-order Volterra and both FFE
variants are also evaluated so that cross-user observations are not reserved
only for neural processing.

Full joint fifth-order Volterra exceeds the declared 1,000-feature budget
and is recorded as not evaluated, not as a failed equalizer. Finite training
data also affect the much larger joint cubic model. Adding guard bins helps
some points but does not eliminate all cross-user distortion in the shared
MZM. The training-only front-end channel estimate is frozen for validation
and test; post-equalizers do not adapt to test payload targets.

This study models localized DFT spreading, individual RF distortion and a
shared MZM. It does not recreate the paper's optical/wireless hardware,
per-user bit rates or full fiber propagation. Larger joint observation
windows and matched-capacity studies are follow-ups, not hidden assumptions.
