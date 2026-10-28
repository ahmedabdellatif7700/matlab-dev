# MZM study findings

At 64-QAM, no extra echo and drive index 1.0, pooled BER over 269,982 test
bits is 0.005260 for complex FFE, 0.001456 for fifth-order Volterra and
0.001259 for the multilevel NN. Both nonlinear models pass the 3.8e-3
reference while FFE does not. This demonstrates the value of nonlinear
compensation, but does not establish that an NN is the only sufficient model.

At drive index 1.8 the corresponding BERs are approximately 0.1274, 0.1201
and 0.1206. Strong MZM folding limits every tested receiver. At low drive,
fixed receiver noise similarly limits the signal; the sweep does not hide
that penalty by normalizing noise to received power.

The multilevel NN can have much lower EVM than a smooth regression model
because its output is attracted to known QAM levels. That effect is not the
same as a proportional BER improvement. Fixed-alpha and adaptive-alpha
variants do not have a uniform BER ranking. Their optimizer and architecture
differ from Liu's SGD experiment, so its convergence counts or sensitivity
gains are not claimed reproduced.

These are two-complex-tap, finite-training comparisons. Longer observation
windows, independent hyperparameter studies, measured device characteristics
and additional test bits are sensible follow-ups before drawing deployment
conclusions.
