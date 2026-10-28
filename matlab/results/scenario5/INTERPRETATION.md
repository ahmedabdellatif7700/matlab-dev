# Wideband OFDM study findings

The measured transmitter PAPR distribution is higher for OFDM than for the
matched-bandwidth RRC single-carrier reference. These are PAPR diagnostics,
not BER comparisons through different channels.

At 64-QAM and 6 dB backoff, pilot-LS front-end BER is approximately 0.01721;
the known-linear-shape diagnostic reduces it to 0.01417. Neither reaches the
3.8e-3 reference. The multilevel NN similarly changes from approximately
0.01914 to 0.01542. Better linear channel knowledge helps, but does not remove
the remaining nonlinear ICI. All comparisons use the same pilots and do not
fit to payload test targets.

The local NN uses only the current and two neighboring frequency carriers.
OFDM nonlinear mixing can involve the entire block. Its failure to reach
the 64-QAM target is not evidence that every neural OFDM receiver must fail.
Conversely, reduced EVM from multilevel output does not by itself imply a
better BER. Several 16-QAM/backoff conditions are already handled by simpler
receivers within the reference target.

This is a 28 GHz, ideal-SSB 10 km equivalent link with illustrative RF
distortion, fixed receiver noise and finite pilots. Optical dBm, measured
amplifier characteristics and the experimental paper's numerical curves are
not calibrated. A future full-block or time-domain receiver should be tested
against the same waveform data and matched non-neural observation budgets.
