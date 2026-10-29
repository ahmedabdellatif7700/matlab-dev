function rx = recover(cfg,detector,tx)
%RECOVER Matched RRC, prefix-only timing calibration and symbol sampling.
baseband = detector.baseband;
h = rcosdesign(cfg.rrc_rolloff,cfg.rrc_span_symbols,tx.internal_sps,"sqrt").';
matched = fftfilt(h,[baseband; zeros(numel(h)-1,1)]);
% Sum of Tx and Rx RRC delays, expressed on the internal grid.
origin = cfg.rrc_span_symbols*tx.internal_sps+1;
keep = cfg.guard_symbols+(1:cfg.training_symbols+cfg.payload_symbols).';
centers = origin+(keep-1)*tx.internal_sps;
reference = tx.symbols(keep);
calibration = 1:cfg.training_symbols;
interpolator = spline((1:numel(matched)).',matched);
best_error = Inf; best_offset = 0; best_gain = 1;
for offset = linspace(-0.05*tx.internal_sps,0.05*tx.internal_sps,401)
    samples = reshape(ppval(interpolator,centers(calibration)+offset),[],1);
    gain = (samples'*reference(calibration))/(samples'*samples);
    error = mean(abs(gain*samples-reference(calibration)).^2);
    if error < best_error
        best_error = error; best_offset = offset; best_gain = gain;
    end
end
sampled = reshape(ppval(interpolator,centers+best_offset),[],1)*best_gain;
payload = cfg.training_symbols+(1:cfg.payload_symbols);
rx = detector;
rx.sampled = sampled; rx.reference = reference; rx.payload_indices = payload;
rx.timing_offset_samples = best_offset; rx.calibration_gain = best_gain;
end
