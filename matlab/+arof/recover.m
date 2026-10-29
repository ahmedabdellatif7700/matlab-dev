function rx = recover(cfg,detector,tx)
%RECOVER Matched RRC, prefix-only timing calibration and symbol sampling.
baseband = detector.baseband;
h = rcosdesign(cfg.rrc_rolloff,cfg.rrc_span_symbols,tx.internal_sps,"sqrt").';
matched = conv(baseband,h);
% Sum of Tx and Rx RRC delays, expressed on the internal grid.
origin = cfg.rrc_span_symbols*tx.internal_sps+1;
keep = cfg.guard_symbols+(1:cfg.training_symbols+cfg.payload_symbols).';
centers = origin+(keep-1)*tx.internal_sps;
reference = tx.symbols(keep);
calibration = 1:cfg.training_symbols;
best_error = Inf; best_offset = 0; best_gain = 1;
for offset = linspace(-0.05*tx.internal_sps,0.05*tx.internal_sps,401)
    samples = interp1(matched,centers(calibration)+offset,"spline");
    gain = (samples'*reference(calibration))/(samples'*samples);
    error = mean(abs(gain*samples-reference(calibration)).^2);
    if error < best_error
        best_error = error; best_offset = offset; best_gain = gain;
    end
end
sampled = interp1(matched,centers+best_offset,"spline")*best_gain;
payload = cfg.training_symbols+(1:cfg.payload_symbols);
rx = detector;
rx.sampled = sampled; rx.reference = reference; rx.payload_indices = payload;
rx.timing_offset_samples = best_offset; rx.calibration_gain = best_gain;
end
