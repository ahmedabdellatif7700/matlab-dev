# Coupled RF/optical study findings

At 1.02 km, nonlinear RF drive and DSB transport, complex FFE, fifth-order
Volterra and the multilevel NN have pooled BERs of approximately 0.4605,
0.4646 and 0.4627. The deep dispersion-fading null and resulting low SNR
dominate this condition; extra neural nonlinearity does not recover the lost
information with the specified observation window.

For SSB at the same length and RF operating point, all three have zero
observed errors over 179,976 test bits. That is a finite-sample observation,
not proof of zero BER. Their EVMs differ substantially (approximately
7.839%, 4.570% and 0.125%) even though the BER result is the same. SSB's
removal of magnitude fading is a link-design benefit, not an NN benefit.

At lengths where DSB changes sign, uncompensated constellations can rotate
by pi. The complex FFE already handles that rotation; it should not be
presented as uniquely neural phase recovery. Both SSB and DSB assume the
known bulk propagation phase/delay has been removed.

The modulator is linearized in this scenario. The model does not claim to
propagate Scenario 2's large-signal optical sidebands through a full
nonlinear fiber/photodetector simulation. Three-tap buffering, noise,
training budgets and excluded impairments are specified in SCENARIOS.md.
